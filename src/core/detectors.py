"""Rule-based intent detection and identity parsing.

Deliberately simple and deterministic: routing happens before any LLM call, so it
is cheap, testable, and works even when the LLM is disabled.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


def normalize(text: str) -> str:
    """Lowercase and strip accents, so 'Données' and 'donnees' match."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class PersonalDataDetector:
    """Detects questions about the user's own contracts, quotes or claims."""

    KEYWORDS = (
        "mes contrats", "mon contrat", "mes devis", "mon devis", "mes sinistres", "mon sinistre",
        "mes informations", "mon historique", "mes donnees", "mon profil", "mes polices",
        "ma police", "mon assurance", "mes garanties", "mon compte", "mes paiements",
        "ma prime", "mes primes",
    )

    def needs_personal_data(self, query: str) -> bool:
        text = normalize(query)
        return any(keyword in text for keyword in self.KEYWORDS)


class QuotationDetector:
    """Detects requests for a car-insurance quote, not questions about pricing in general."""

    DEFINITION_MARKERS = (
        "qu'est-ce", "qu est-ce", "qu'est ce", "c'est quoi", "definition", "definir", "explique",
        "signifie", "que veut dire", "comment ca marche", "comment fonctionne", "a quoi sert",
    )
    QUOTE_PHRASES = (
        "je veux un devis", "j'ai besoin d'un devis", "donnez-moi un devis", "donne-moi un devis",
        "faire un devis", "devis pour", "un devis auto", "calculer le prix", "estimer le cout",
        "combien coute", "combien va me couter", "quel est le prix", "tarif pour",
    )
    PRICE_WORDS = ("devis", "prix", "tarif", "cout", "estimation", "combien")
    VEHICLE_WORDS = ("auto", "voiture", "vehicule", "automobile", "moto")
    PRODUCT_TYPES = {
        "auto": ("auto", "voiture", "vehicule", "automobile"),
        "vie": ("assurance vie", "epargne", "capitalisation"),
        "habitation": ("habitation", "maison", "appartement", "logement"),
        "sante": ("sante", "medical", "hospitalisation"),
        "prevoyance": ("prevoyance", "invalidite", "deces"),
    }

    def needs_quotation(self, query: str) -> bool:
        text = normalize(query)
        if any(marker in text for marker in self.DEFINITION_MARKERS):
            return False
        if any(phrase in text for phrase in self.QUOTE_PHRASES):
            return True
        has_price = any(word in text for word in self.PRICE_WORDS)
        has_vehicle = any(word in text for word in self.VEHICLE_WORDS)
        return has_price and has_vehicle

    def product_type(self, query: str) -> Optional[str]:
        text = normalize(query)
        for product, words in self.PRODUCT_TYPES.items():
            if any(word in text for word in words):
                return product
        return None


@dataclass(frozen=True)
class Identity:
    nom: Optional[str] = None
    prenom: Optional[str] = None
    telephone: Optional[str] = None  # 8 digits, without the +216 prefix

    @property
    def complete(self) -> bool:
        return bool(self.nom and self.prenom and self.telephone)


_PHONE_RE = re.compile(r"(?:\+|00)?\d[\d\s.\-()]{6,}\d")
_EMAIL_RE = re.compile(r"\S+@\S+")
_LABELLED_RE = {
    "prenom": re.compile(r"pr[ée]nom\s*(?:est|:|=)?\s*([^\d,;:\n]+?)(?=\s+(?:nom|t[ée]l|num)|[,;\n]|$)", re.I),
    "nom": re.compile(r"(?<!pr[ée])(?<!pre)\bnom\s*(?:est|:|=)?\s*([^\d,;:\n]+?)(?=\s+(?:pr[ée]nom|t[ée]l|num)|[,;\n]|$)", re.I),
}
_NAME_STOPWORDS = {
    "je", "j", "m'appelle", "appelle", "suis", "moi", "c'est", "mon", "ma", "mes", "nom", "prenom",
    "telephone", "tel", "numero", "num", "est", "et", "le", "la", "voici", "bonjour", "veux",
    "voudrais", "voir", "avoir", "contrat", "contrats", "devis", "sinistre", "sinistres",
    "informations", "donnees", "profil", "svp", "merci", "s'il", "plait", "vous",
}
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]*")


def parse_phone(text: str) -> Optional[str]:
    for match in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if digits.startswith("00216"):
            digits = digits[5:]
        elif digits.startswith("216") and len(digits) == 11:
            digits = digits[3:]
        if len(digits) == 8:
            return digits
    return None


def parse_identity(text: str) -> Identity:
    """Extract first name, last name and phone number from a free-text answer.

    Accepts "nom: Ben Ali, prénom: Mariem, tél: 98 765 432" as well as the short
    form "Mariem Ben Ali 98765432" (first word = first name, the rest = last name).
    """
    telephone = parse_phone(text)
    without_contact = _EMAIL_RE.sub(" ", _PHONE_RE.sub(" ", text))

    prenom_match = _LABELLED_RE["prenom"].search(without_contact)
    nom_match = _LABELLED_RE["nom"].search(without_contact)
    if prenom_match or nom_match:
        prenom = prenom_match.group(1).strip() if prenom_match else None
        nom = nom_match.group(1).strip() if nom_match else None
        return Identity(nom=nom or None, prenom=prenom or None, telephone=telephone)

    words = [w for w in _WORD_RE.findall(without_contact) if normalize(w) not in _NAME_STOPWORDS]
    if len(words) >= 2:
        return Identity(nom=" ".join(words[1:]), prenom=words[0], telephone=telephone)
    return Identity(telephone=telephone)


_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def parse_verification_code(text: str) -> Optional[str]:
    match = _CODE_RE.search(text)
    return match.group(1) if match else None
