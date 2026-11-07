#!/usr/bin/env python3
"""BundKi-Secure-Parser
=====================

Referenzimplementierung einer datenschutzfreundlichen Verarbeitungskette für
amtliche Dokumente (Steuerbescheid, Meldebescheinigung, Aufenthaltstitel):

    Datei -> OCR -> PII-Maskierung (lokal) -> Extraktion -> Plausibilitätsprüfung
          -> Prüfbericht (ohne Klardaten) -> menschliche Entscheidung

Abhängigkeiten (alle lokal betreibbar, keine Cloud-Dienste):
    pip install pytesseract pdf2image pillow
    pip install spacy && python -m spacy download de_core_news_lg   # optional
    Systempakete: tesseract-ocr, tesseract-ocr-deu, poppler-utils

Python >= 3.10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol

logger = logging.getLogger("bundki.secure_parser")


# ---------------------------------------------------------------------------
# 1. Konfiguration & Fehlerklassen
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParserConfig:
    """Zentrale, unveränderliche Konfiguration."""

    ocr_language: str = "deu"
    ocr_dpi: int = 300
    max_pages: int = 20
    max_file_size_mb: int = 25
    min_ocr_confidence: float = 60.0  # Tesseract-Skala 0-100
    use_spacy_ner: bool = True
    spacy_model: str = "de_core_news_lg"
    allowed_suffixes: frozenset[str] = frozenset(
        {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )


class BundKiError(Exception):
    """Basisklasse aller fachlichen Fehler."""


class DocumentLoadError(BundKiError):
    """Datei fehlt, ist zu groß, hat ein unzulässiges Format oder ist defekt."""


class OCRError(BundKiError):
    """OCR-Engine nicht verfügbar oder Erkennung fehlgeschlagen."""


class AnonymizationError(BundKiError):
    """Maskierung fehlgeschlagen. Verarbeitung MUSS abgebrochen werden (fail closed)."""


# ---------------------------------------------------------------------------
# 2. OCR-Schicht
# ---------------------------------------------------------------------------
@dataclass
class OCRResult:
    text: str
    mean_confidence: float
    page_count: int


class OCREngine:
    """Lokale OCR auf Basis von Tesseract (kein Datenabfluss)."""

    def __init__(self, config: ParserConfig) -> None:
        self._cfg = config

    def _validate_file(self, path: Path) -> None:
        if not path.is_file():
            raise DocumentLoadError("Datei nicht gefunden.")
        if path.suffix.lower() not in self._cfg.allowed_suffixes:
            raise DocumentLoadError(f"Dateityp {path.suffix!r} nicht zulässig.")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self._cfg.max_file_size_mb:
            raise DocumentLoadError(f"Datei zu groß ({size_mb:.1f} MB).")

    def _load_images(self, path: Path) -> list[Any]:
        try:
            if path.suffix.lower() == ".pdf":
                from pdf2image import convert_from_path

                return convert_from_path(
                    str(path), dpi=self._cfg.ocr_dpi, last_page=self._cfg.max_pages
                )
            from PIL import Image, ImageSequence

            with Image.open(path) as img:
                frames = [f.copy() for f in ImageSequence.Iterator(img)]
            return frames[: self._cfg.max_pages]
        except ImportError as exc:
            raise OCRError(f"Fehlende Abhängigkeit: {exc.name}") from exc
        except Exception as exc:  # defekte PDFs/Bilder
            raise DocumentLoadError("Dokument konnte nicht geladen werden.") from exc

    @staticmethod
    def _preprocess(image: Any) -> Any:
        """Graustufen + Kontrastnormalisierung verbessert die Trefferquote."""
        from PIL import ImageOps

        return ImageOps.autocontrast(ImageOps.grayscale(image))

    def read(self, path: Path) -> OCRResult:
        self._validate_file(path)
        images = self._load_images(path)
        if not images:
            raise DocumentLoadError("Dokument enthält keine Seiten.")
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:
            raise OCRError("pytesseract ist nicht installiert.") from exc

        lines_out: list[str] = []
        confidences: list[float] = []
        for page_no, image in enumerate(images, start=1):
            try:
                data = pytesseract.image_to_data(
                    self._preprocess(image),
                    lang=self._cfg.ocr_language,
                    config="--oem 1 --psm 3",
                    output_type=Output.DICT,
                )
            except Exception as exc:
                raise OCRError(f"OCR auf Seite {page_no} fehlgeschlagen.") from exc

            grouped: dict[tuple[int, int, int], list[str]] = {}
            for i, word in enumerate(data["text"]):
                word = word.strip()
                if not word:
                    continue
                conf = float(data["conf"][i])
                if conf >= 0:
                    confidences.append(conf)
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                grouped.setdefault(key, []).append(word)
            lines_out.extend(" ".join(words) for _, words in sorted(grouped.items()))

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        logger.info("OCR abgeschlossen: %d Seite(n), Konfidenz %.1f", len(images), mean_conf)
        return OCRResult("\n".join(lines_out), mean_conf, len(images))


# ---------------------------------------------------------------------------
# 3. Anonymisierung / Pseudonymisierung (DSGVO Art. 4 Nr. 5, Art. 25, Art. 32)
# ---------------------------------------------------------------------------
class PIIType(str, Enum):
    PERSON = "PERSON"
    E_MAIL = "E_MAIL"
    IBAN = "IBAN"
    STEUER_ID = "STEUER_ID"
    STEUERNUMMER = "STEUERNUMMER"
    TELEFON = "TELEFON"
    GEBURTSDATUM = "GEBURTSDATUM"
    ADRESSE = "ADRESSE"
    AUSWEIS_NR = "AUSWEIS_NR"


@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    pii_type: PIIType
    score: float


def valid_iban(value: str) -> bool:
    """IBAN-Prüfsumme nach ISO 13616 (Modulo 97)."""
    iban = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def valid_steuer_id(value: str) -> bool:
    """Prüfziffer der steuerlichen Identifikationsnummer (ISO 7064, MOD 11,10)."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 11 or digits[0] == "0":
        return False
    product = 10
    for ch in digits[:10]:
        total = (int(ch) + product) % 10
        total = 10 if total == 0 else total
        product = (total * 2) % 11
    check = 11 - product
    check = 0 if check == 10 else check
    return check == int(digits[10])


@dataclass(frozen=True)
class PatternRecognizer:
    """Regex-Erkenner; Validatoren steuern den Score (Maskierung erfolgt immer)."""

    pii_type: PIIType
    pattern: re.Pattern[str]
    score: float
    validator: Optional[Callable[[str], bool]] = None
    invalid_score: float = 0.5

    def find(self, text: str) -> Iterator[Detection]:
        grp = 1 if self.pattern.groups else 0
        for m in self.pattern.finditer(text):
            if m.group(grp) is None:
                continue
            value = m.group(grp)
            ok = self.validator(value) if self.validator else True
            yield Detection(
                m.start(grp), m.end(grp), self.pii_type,
                self.score if ok else self.invalid_score,
            )


_NAME = r"[A-ZÄÖÜ][\wäöüß\-]+(?:[ \t]+[A-ZÄÖÜ][\wäöüß\-]+){0,3}"
_STREETS = r"(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer)"


def _default_recognizers() -> list[PatternRecognizer]:
    R = PatternRecognizer
    return [
        R(PIIType.E_MAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 0.99),
        R(PIIType.IBAN, re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b"),
          0.95, valid_iban, 0.7),
        R(PIIType.STEUER_ID, re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
          0.95, valid_steuer_id, 0.6),
        R(PIIType.STEUERNUMMER, re.compile(r"\b\d{2,3}/\d{3}/\d{4,5}\b"), 0.9),
        R(PIIType.TELEFON,
          re.compile(r"(?<![\w.])(?:\+49|0049|0)[\s/\-]?\(?\d{2,5}\)?[\s/\-]?\d{3,}(?:[\s/\-]?\d+)*"),
          0.5),
        R(PIIType.GEBURTSDATUM,
          re.compile(r"(?i)(?:geboren am|geburtsdatum|geb\.)\s*:?\s*(\d{1,2}\.\d{1,2}\.\d{4})"), 0.95),
        R(PIIType.ADRESSE,
          re.compile(rf"\b[A-ZÄÖÜ][\wäöüß\-]*{_STREETS}\s*\d{{1,4}}\s?[a-z]?\b"), 0.85),
        R(PIIType.ADRESSE,
          re.compile(r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+(?:[ \-][A-ZÄÖÜ][a-zäöüß]+)*"), 0.7),
        R(PIIType.PERSON,
          re.compile(rf"(?i:\b(?:vorname|familienname|nachname|geburtsname|name)\s*:\s*)({_NAME})"), 0.85),
        R(PIIType.PERSON, re.compile(rf"\b(?:Herrn?|Frau)\s+((?:Dr\.\s+)?{_NAME})"), 0.8),
        R(PIIType.AUSWEIS_NR,
          re.compile(r"(?i)(?:AZR-Nummer|eAT-Nr\.?|Dokumentennummer|Passnummer)\s*:?\s*([A-Z0-9]{6,12})\b"), 0.9),
    ]


class PseudonymVault:
    """Hält die Zuordnung Token <-> Originalwert ausschließlich im Arbeitsspeicher.

    Wird nie geloggt, nie serialisiert und nach der Verarbeitung gelöscht.
    """

    def __init__(self) -> None:
        self._forward: dict[tuple[PIIType, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[PIIType, int] = {}

    def token_for(self, pii_type: PIIType, value: str) -> str:
        key = (pii_type, value.strip())
        if key not in self._forward:
            self._counters[pii_type] = self._counters.get(pii_type, 0) + 1
            token = f"[{pii_type.value}_{self._counters[pii_type]}]"
            self._forward[key] = token
            self._reverse[token] = value.strip()
        return self._forward[key]

    def values_in(self, masked_text: str, pii_type: PIIType) -> list[str]:
        """Löst Tokens eines Typs lokal auf (nur für die Validierung)."""
        tokens = re.findall(rf"\[{pii_type.value}_\d+\]", masked_text)
        return [self._reverse[t] for t in dict.fromkeys(tokens) if t in self._reverse]

    def clear(self) -> None:
        self._forward.clear()
        self._reverse.clear()
        self._counters.clear()


@dataclass
class AnonymizationResult:
    masked_text: str
    stats: dict[str, int]
    vault: PseudonymVault = field(repr=False)


class Anonymizer:
    """Hybride PII-Erkennung: Regeln + Prüfsummen + optionales lokales spaCy-NER."""

    def __init__(self, config: ParserConfig) -> None:
        self._recognizers = _default_recognizers()
        self._nlp: Any = None
        if config.use_spacy_ner:
            try:
                import spacy

                self._nlp = spacy.load(config.spacy_model, disable=["lemmatizer"])
                logger.info("spaCy-NER geladen (%s).", config.spacy_model)
            except Exception:
                logger.warning("spaCy-Modell nicht verfügbar - nur regelbasierte Erkennung aktiv.")

    def _ner_detections(self, text: str) -> Iterator[Detection]:
        if self._nlp is None:
            return
        for ent in self._nlp(text).ents:
            if ent.label_ == "PER" and len(ent.text) >= 3:
                yield Detection(ent.start_char, ent.end_char, PIIType.PERSON, 0.6)

    @staticmethod
    def _resolve_overlaps(found: list[Detection]) -> list[Detection]:
        accepted: list[Detection] = []
        for det in sorted(found, key=lambda d: (-d.score, -(d.end - d.start), d.start)):
            if all(det.end <= a.start or det.start >= a.end for a in accepted):
                accepted.append(det)
        return sorted(accepted, key=lambda d: d.start)

    def anonymize(self, text: str) -> AnonymizationResult:
        try:
            found: list[Detection] = []
            for rec in self._recognizers:
                found.extend(rec.find(text))
            found.extend(self._ner_detections(text))
            final = self._resolve_overlaps(found)

            vault = PseudonymVault()
            parts: list[str] = []
            cursor = 0
            stats: dict[str, int] = {}
            for det in final:
                parts.append(text[cursor:det.start])
                parts.append(vault.token_for(det.pii_type, text[det.start:det.end]))
                cursor = det.end
                stats[det.pii_type.value] = stats.get(det.pii_type.value, 0) + 1
            parts.append(text[cursor:])
            return AnonymizationResult("".join(parts), stats, vault)
        except Exception as exc:
            # Fail closed: lieber abbrechen als unmaskierten Text weiterreichen.
            raise AnonymizationError("Maskierung fehlgeschlagen.") from exc


# ---------------------------------------------------------------------------
# 4. Dokumentklassifikation & Informationsextraktion
# ---------------------------------------------------------------------------
class DocumentType(str, Enum):
    STEUERBESCHEID = "steuerbescheid"
    MELDEBESCHEINIGUNG = "meldebescheinigung"
    AUFENTHALTSTITEL = "aufenthaltstitel"
    UNBEKANNT = "unbekannt"


_KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.STEUERBESCHEID: (
        "steuerbescheid", "einkommensteuer", "finanzamt", "festsetzung", "zu versteuerndes einkommen"),
    DocumentType.MELDEBESCHEINIGUNG: (
        "meldebescheinigung", "einwohnermeldeamt", "bürgeramt", "anmeldung", "hauptwohnung", "wohnsitz"),
    DocumentType.AUFENTHALTSTITEL: (
        "aufenthaltstitel", "aufenthaltserlaubnis", "niederlassungserlaubnis", "ausländerbehörde",
        "aufenthg", "elektronischer aufenthaltstitel"),
}

_DATE = r"(\d{1,2}\.\d{1,2}\.\d{4})"
_ISSUE_RE = re.compile(r"(?i)(?:ausstellungsdatum|ausgestellt am|datum des bescheids?|bescheid vom)\s*:?\s*" + _DATE)
_VALID_RE = re.compile(r"(?i)(?:gültig bis|gueltig bis|befristet bis)\s*:?\s*" + _DATE)
_REG_RE = re.compile(r"(?i)(?:anmeldedatum|angemeldet am|einzugsdatum|zuzugsdatum)\s*:?\s*" + _DATE)
_AMOUNT_RE = re.compile(
    r"(?i)(festgesetzte einkommensteuer|zu versteuerndes einkommen|nachzahlung|erstattung)"
    r"[^\d\-\n]{0,40}(-?\d{1,3}(?:\.\d{3})*,\d{2})"
)


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%d.%m.%Y").date()
    except ValueError:
        return None


def parse_amount(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value.replace(".", "").replace(",", "."))
    except InvalidOperation:
        return None


@dataclass
class ExtractedFields:
    """Strukturierte Daten. Sensible Felder werden nie in den Bericht übernommen."""

    document_type: DocumentType
    issue_date: Optional[date] = None
    valid_until: Optional[date] = None
    registration_date: Optional[date] = None
    amounts: dict[str, Decimal] = field(default_factory=dict)
    # --- nur lokal für die Validierung, nicht exportierbar ---
    birth_date: Optional[date] = field(default=None, repr=False)
    steuer_ids: list[str] = field(default_factory=list, repr=False)
    ibans: list[str] = field(default_factory=list, repr=False)
    has_steuernummer: bool = False

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type.value,
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "registration_date": self.registration_date.isoformat() if self.registration_date else None,
            "amounts_eur": {k: str(v) for k, v in self.amounts.items()},
        }


class FieldExtractor:
    """Arbeitet auf dem MASKIERTEN Text; sensible Werte nur über den lokalen Vault."""

    @staticmethod
    def classify(masked_text: str) -> DocumentType:
        lowered = masked_text.lower()
        scores = {dt: sum(lowered.count(k) for k in kws) for dt, kws in _KEYWORDS.items()}
        best, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best if best_score >= 2 else DocumentType.UNBEKANNT

    def extract(self, masked_text: str, vault: PseudonymVault) -> ExtractedFields:
        fields = ExtractedFields(document_type=self.classify(masked_text))

        def first_date(rx: re.Pattern[str]) -> Optional[date]:
            m = rx.search(masked_text)
            return parse_date(m.group(1)) if m else None

        fields.issue_date = first_date(_ISSUE_RE)
        fields.valid_until = first_date(_VALID_RE)
        fields.registration_date = first_date(_REG_RE)

        for label, raw in _AMOUNT_RE.findall(masked_text):
            amount = parse_amount(raw)
            if amount is not None:
                fields.amounts.setdefault(label.lower(), amount)

        births = vault.values_in(masked_text, PIIType.GEBURTSDATUM)
        fields.birth_date = parse_date(births[0]) if births else None
        fields.steuer_ids = vault.values_in(masked_text, PIIType.STEUER_ID)
        fields.ibans = vault.values_in(masked_text, PIIType.IBAN)
        fields.has_steuernummer = bool(vault.values_in(masked_text, PIIType.STEUERNUMMER))
        return fields


# ---------------------------------------------------------------------------
# 5. Plausibilitätsprüfung
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str


class PlausibilityChecker:
    """Regelbasierte, nachvollziehbare Prüfungen (erklärbar, deterministisch)."""

    MAX_AGE_YEARS = 120

    def __init__(self, config: ParserConfig, today: Callable[[], date] = date.today) -> None:
        self._cfg = config
        self._today = today

    def check(self, f: ExtractedFields, ocr_confidence: float) -> list[Finding]:
        out: list[Finding] = []
        today = self._today()
        add = lambda c, s, m: out.append(Finding(c, s, m))  # noqa: E731

        if ocr_confidence < self._cfg.min_ocr_confidence:
            add("V002", Severity.WARNING, f"OCR-Konfidenz niedrig ({ocr_confidence:.0f}).")
        if f.document_type is DocumentType.UNBEKANNT:
            add("V001", Severity.WARNING, "Dokumenttyp konnte nicht sicher bestimmt werden.")

        # Personenbezogene Plausibilität
        if f.birth_date:
            if f.birth_date > today:
                add("V011", Severity.ERROR, "Geburtsdatum liegt in der Zukunft.")
            elif (today - f.birth_date).days / 365.25 > self.MAX_AGE_YEARS:
                add("V012", Severity.ERROR, "Geburtsdatum ergibt unplausibles Alter.")
        elif f.document_type in (DocumentType.MELDEBESCHEINIGUNG, DocumentType.AUFENTHALTSTITEL):
            add("V010", Severity.WARNING, "Kein Geburtsdatum erkannt.")

        for sid in f.steuer_ids:
            if not valid_steuer_id(sid):
                add("V020", Severity.ERROR, "Steuer-ID besteht die Prüfziffernkontrolle nicht.")
        for iban in f.ibans:
            if not valid_iban(iban):
                add("V040", Severity.ERROR, "IBAN besteht die Prüfsummenkontrolle nicht.")

        # Zeitliche Konsistenz
        if f.issue_date and f.issue_date > today:
            add("V030", Severity.ERROR, "Ausstellungsdatum liegt in der Zukunft.")
        if f.issue_date and f.valid_until and f.valid_until < f.issue_date:
            add("V031", Severity.ERROR, "Gültigkeitsende liegt vor dem Ausstellungsdatum.")
        if f.issue_date and f.registration_date and f.registration_date > f.issue_date:
            add("V033", Severity.ERROR, "Anmeldedatum liegt nach dem Ausstellungsdatum.")
        if f.birth_date and f.issue_date and f.birth_date > f.issue_date:
            add("V013", Severity.ERROR, "Geburtsdatum liegt nach dem Ausstellungsdatum.")

        # Dokumenttyp-spezifisch
        if f.document_type is DocumentType.AUFENTHALTSTITEL:
            if f.valid_until is None:
                add("V034", Severity.WARNING, "Kein Gültigkeitsende erkannt.")
            elif f.valid_until < today:
                add("V032", Severity.WARNING, "Aufenthaltstitel ist abgelaufen.")
        if f.document_type is DocumentType.STEUERBESCHEID:
            if not f.steuer_ids and not f.has_steuernummer:
                add("V050", Severity.WARNING, "Weder Steuer-ID noch Steuernummer erkannt.")
            if not f.amounts:
                add("V051", Severity.WARNING, "Keine Beträge erkannt.")
        return out


# ---------------------------------------------------------------------------
# 6. Pipeline & Prüfbericht
# ---------------------------------------------------------------------------
class MaskedTextAnalyzer(Protocol):
    """Erweiterungspunkt, z. B. für ein lokal gehostetes LLM.

    Erhält ausschließlich maskierten Text (Privacy Boundary).
    """

    def analyze(self, masked_text: str) -> dict[str, Any]: ...


@dataclass
class ProcessingReport:
    """Bericht ohne Klardaten (datenschutzfreundlich archivierbar)."""

    document_hash: str
    processed_at: str
    ocr_confidence: float
    pages: int
    pii_statistics: dict[str, int]
    fields: dict[str, Any]
    findings: list[dict[str, str]]
    requires_human_review: bool
    masked_text: Optional[str] = None
    analyzer_output: Optional[dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class SecureParserPipeline:
    def __init__(
        self,
        config: Optional[ParserConfig] = None,
        analyzer: Optional[MaskedTextAnalyzer] = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._cfg = config or ParserConfig()
        self._ocr = OCREngine(self._cfg)
        self._anonymizer = Anonymizer(self._cfg)
        self._extractor = FieldExtractor()
        self._checker = PlausibilityChecker(self._cfg, today)
        self._analyzer = analyzer

    def process_file(self, path: Path, include_masked_text: bool = False) -> ProcessingReport:
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "n/a"
        logger.info("Verarbeitung gestartet (sha256=%s…)", digest[:12])
        ocr = self._ocr.read(path)
        return self.process_text(ocr.text, digest, ocr.mean_confidence, ocr.page_count, include_masked_text)

    def process_text(
        self,
        raw_text: str,
        document_hash: str = "n/a",
        ocr_confidence: float = 100.0,
        pages: int = 1,
        include_masked_text: bool = False,
    ) -> ProcessingReport:
        anon = self._anonymizer.anonymize(raw_text)
        del raw_text  # Klartext verlässt den Gültigkeitsbereich der Maskierungsstufe
        try:
            fields = self._extractor.extract(anon.masked_text, anon.vault)
            findings = self._checker.check(fields, ocr_confidence)
            analyzer_out = self._analyzer.analyze(anon.masked_text) if self._analyzer else None
        finally:
            anon.vault.clear()  # Datenminimierung: Zuordnungstabelle sofort vernichten

        needs_review = any(f.severity is not Severity.INFO for f in findings)
        logger.info("Fertig: Typ=%s, Befunde=%d, Prüfung nötig=%s",
                    fields.document_type.value, len(findings), needs_review)
        return ProcessingReport(
            document_hash=document_hash,
            processed_at=datetime.now().isoformat(timespec="seconds"),
            ocr_confidence=round(ocr_confidence, 1),
            pages=pages,
            pii_statistics=anon.stats,
            fields=fields.to_safe_dict(),
            findings=[{"code": f.code, "severity": f.severity.value, "message": f.message} for f in findings],
            requires_human_review=needs_review,
            masked_text=anon.masked_text if include_masked_text else None,
            analyzer_output=analyzer_out,
        )


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
_DEMO_TEXT = """Stadt Musterstadt - Bürgeramt
Meldebescheinigung (Anmeldung Hauptwohnung)
Name: Erika Mustermann
Geburtsdatum: 12.08.1985
Anschrift: Musterstraße 12, 80331 München
Steuer-ID: 65929970489
IBAN: DE89 3704 0044 0532 0130 00
Telefon: 089 1234567, E-Mail: erika.mustermann@example.de
Anmeldedatum: 01.02.2026
Ausstellungsdatum: 15.02.2026
"""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="bundki-secure-parser", description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", type=Path, help="PDF/Bild eines amtlichen Dokuments")
    parser.add_argument("--demo", action="store_true", help="Synthetischen Beispieltext verarbeiten (ohne OCR)")
    parser.add_argument("--show-masked", action="store_true", help="Maskierten Text in den Bericht aufnehmen")
    parser.add_argument("--no-ner", action="store_true", help="spaCy-NER deaktivieren")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    pipeline = SecureParserPipeline(ParserConfig(use_spacy_ner=not args.no_ner))
    try:
        if args.demo:
            report = pipeline.process_text(_DEMO_TEXT, include_masked_text=args.show_masked)
        elif args.file:
            report = pipeline.process_file(args.file, include_masked_text=args.show_masked)
        else:
            parser.error("Datei oder --demo angeben.")
            return 2
    except BundKiError as exc:
        logger.error("Verarbeitung abgebrochen: %s", exc)
        return 1
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())