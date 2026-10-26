"""Car-insurance quotes: extract the vehicle details, validate them, call the quote API.

Extraction uses the LLM (JSON output validated by Pydantic) when it is available,
and falls back to conservative regular expressions otherwise. Values the user did
not give are never invented silently: the answer lists what is missing.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from .detectors import normalize

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    "n_cin", "valeur_venale", "valeur_a_neuf", "date_premiere_mise_en_circulation",
    "puissance", "nombre_place", "classe",
)
FIELD_LABELS = {
    "n_cin": "votre numéro de CIN (8 chiffres)",
    "valeur_venale": "la valeur vénale du véhicule (en DT)",
    "valeur_a_neuf": "la valeur à neuf du véhicule (en DT)",
    "date_premiere_mise_en_circulation": "la date de première mise en circulation",
    "puissance": "la puissance fiscale (CV)",
    "nombre_place": "le nombre de places",
    "classe": "la classe bonus/malus",
}


class VehicleInfo(BaseModel):
    """Parameters of the quote API. Ranges reject obviously wrong extractions."""

    n_cin: str = Field(pattern=r"^\d{8}$")
    valeur_venale: int = Field(ge=1_000, le=2_000_000)
    valeur_a_neuf: int = Field(ge=1_000, le=2_000_000)
    date_premiere_mise_en_circulation: date
    puissance: int = Field(ge=1, le=50)
    nombre_place: int = Field(ge=2, le=9)
    classe: int = Field(ge=1, le=50)
    nature_contrat: Literal["r", "n"] = "r"  # r = responsabilité civile, n = tous risques
    capital_bris_de_glace: int = Field(default=900, ge=0)
    capital_dommage_collision: int = Field(default=60_000, ge=0)

    @field_validator("date_premiere_mise_en_circulation")
    @classmethod
    def not_in_future(cls, value: date) -> date:
        if value > date.today() or value.year < 1950:
            raise ValueError("date de mise en circulation invalide")
        return value

    def as_query_params(self) -> dict[str, Any]:
        params = self.model_dump()
        params["date_premiere_mise_en_circulation"] = self.date_premiere_mise_en_circulation.isoformat()
        return params


class PartialVehicleInfo(BaseModel):
    """What could be read from a message; every field optional."""

    n_cin: Optional[str] = None
    valeur_venale: Optional[int] = None
    valeur_a_neuf: Optional[int] = None
    date_premiere_mise_en_circulation: Optional[str] = None
    puissance: Optional[int] = None
    nombre_place: Optional[int] = None
    classe: Optional[int] = None
    nature_contrat: Optional[Literal["r", "n"]] = None

    def missing(self) -> list[str]:
        return [name for name in REQUIRED_FIELDS if getattr(self, name) in (None, "")]


# --- Regex fallback -----------------------------------------------------------------

_MONEY = r"(\d{1,3}(?:[ . ]\d{3})+|\d{4,7})"


def _to_int(text: str) -> int:
    return int(re.sub(r"\D", "", text))


def _search(pattern: str, text: str) -> Optional[re.Match[str]]:
    return re.search(pattern, text, re.IGNORECASE)


def extract_with_regex(message: str) -> PartialVehicleInfo:
    """Conservative extraction: a value is taken only next to the word that names it."""
    text = normalize(message)
    found: dict[str, Any] = {}

    if m := _search(r"\bcin\b\D{0,15}(\d{8})\b", text):
        found["n_cin"] = m.group(1)
    if m := _search(r"valeur\s+venale\D{0,15}" + _MONEY, text):
        found["valeur_venale"] = _to_int(m.group(1))
    if m := _search(r"valeur\s+a\s+neuf\D{0,15}" + _MONEY, text):
        found["valeur_a_neuf"] = _to_int(m.group(1))
    if "valeur_venale" not in found and (m := _search(r"(?:valeur|vaut|prix)\D{0,15}" + _MONEY, text)):
        found["valeur_venale"] = _to_int(m.group(1))
    if m := _search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        found["date_premiere_mise_en_circulation"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    elif m := _search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text):
        found["date_premiere_mise_en_circulation"] = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    elif m := _search(r"(?:circulation|mise en route|de|depuis|annee|modele)\s+(19[5-9]\d|20\d{2})\b", text):
        found["date_premiere_mise_en_circulation"] = f"{m.group(1)}-01-01"
    if m := _search(r"\b(\d{1,2})\s*(?:cv|chevaux)\b", text) or _search(r"puissance\D{0,10}(\d{1,2})\b", text):
        found["puissance"] = int(m.group(1))
    if m := _search(r"\b(\d)\s*places?\b", text):
        found["nombre_place"] = int(m.group(1))
    if m := _search(r"(?:classe|bonus|malus)\D{0,5}(\d{1,2})\b", text):
        found["classe"] = int(m.group(1))
    if "tous risques" in text or "tous-risques" in text:
        found["nature_contrat"] = "n"
    elif "responsabilite civile" in text or "au tiers" in text:
        found["nature_contrat"] = "r"
    return PartialVehicleInfo(**found)


# --- LLM extraction -----------------------------------------------------------------

LLM_EXTRACTION_PROMPT = """Extrais les informations du véhicule de ce message pour un devis d'assurance auto.
Réponds UNIQUEMENT avec un objet JSON contenant ces clés (null si l'information n'est pas donnée,
ne devine jamais une valeur) :
n_cin (8 chiffres), valeur_venale (entier, DT), valeur_a_neuf (entier, DT),
date_premiere_mise_en_circulation (AAAA-MM-JJ), puissance (CV, entier), nombre_place (entier),
classe (entier), nature_contrat ("r" responsabilité civile ou "n" tous risques).

Message : {message}"""


async def extract_with_llm(llm: Any, message: str) -> PartialVehicleInfo:
    reply = await llm.ainvoke(LLM_EXTRACTION_PROMPT.format(message=message))
    content = getattr(reply, "content", str(reply))
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("pas de JSON dans la réponse du LLM")
    data = {k: v for k, v in json.loads(match.group(0)).items() if v is not None}
    return PartialVehicleInfo(**data)


async def extract_vehicle_info(message: str, llm: Any = None) -> PartialVehicleInfo:
    """LLM first when available; keep any field the regexes found that the LLM missed."""
    from_regex = extract_with_regex(message)
    if llm is None:
        return from_regex
    try:
        from_llm = await extract_with_llm(llm, message)
    except (ValueError, ValidationError, json.JSONDecodeError) as error:
        logger.warning("LLM extraction failed, using regex result: %s", type(error).__name__)
        return from_regex
    merged = from_regex.model_dump()
    merged.update({k: v for k, v in from_llm.model_dump().items() if v is not None})
    return PartialVehicleInfo(**merged)


def missing_fields_message(missing: list[str]) -> str:
    items = "\n".join(f"- {FIELD_LABELS[name]}" for name in missing)
    return (
        "Pour calculer votre devis auto, il me manque :\n"
        f"{items}\n\n"
        "Vous pouvez me les écrire ici, ou utiliser le bouton « Devis Auto »."
    )


# --- API client ---------------------------------------------------------------------

class QuotationError(RuntimeError):
    pass


class QuotationClient:
    """Calls the external quote API. Logs only the outcome, never the parameters (CIN)."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout

    async def quote(self, vehicle: VehicleInfo) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._url, params=vehicle.as_query_params())
        except httpx.HTTPError as error:
            logger.error("Quote API unreachable: %s", type(error).__name__)
            raise QuotationError("Le service de devis est injoignable pour le moment.") from error
        if response.status_code != 200:
            logger.error("Quote API returned HTTP %s", response.status_code)
            raise QuotationError(f"Le service de devis a répondu avec une erreur ({response.status_code}).")
        return response.json()
