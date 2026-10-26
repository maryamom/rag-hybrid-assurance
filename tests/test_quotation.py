from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from src.core.quotation import VehicleInfo, extract_vehicle_info, extract_with_regex


def test_regex_takes_values_only_next_to_their_label():
    info = extract_with_regex(
        "Devis auto : CIN 12345678, valeur vénale 45 000 DT, valeur à neuf 52000, "
        "mise en circulation 2019, 6 CV, 5 places, classe 3, tous risques"
    )
    assert info.n_cin == "12345678"
    assert info.valeur_venale == 45000
    assert info.valeur_a_neuf == 52000
    assert info.date_premiere_mise_en_circulation == "2019-01-01"
    assert (info.puissance, info.nombre_place, info.classe) == (6, 5, 3)
    assert info.nature_contrat == "n"
    assert info.missing() == []


def test_regex_does_not_turn_a_price_into_a_year():
    # The old code read "valeur 45000" as the year 4500.
    info = extract_with_regex("valeur 45000")
    assert info.valeur_venale == 45000
    assert info.date_premiere_mise_en_circulation is None


def test_regex_does_not_take_any_8_digit_number_as_cin():
    info = extract_with_regex("mon numéro est 98765432")
    assert info.n_cin is None


def test_missing_fields_are_reported_not_invented():
    info = extract_with_regex("je veux un devis pour ma voiture de 6 CV")
    assert "n_cin" in info.missing()
    assert "valeur_venale" in info.missing()


def test_vehicle_validation_rejects_bad_values():
    with pytest.raises(ValidationError):
        VehicleInfo(
            n_cin="1234",  # not 8 digits
            valeur_venale=45000, valeur_a_neuf=52000,
            date_premiere_mise_en_circulation=date(2019, 1, 1),
            puissance=6, nombre_place=5, classe=3,
        )
    with pytest.raises(ValidationError):
        VehicleInfo(
            n_cin="12345678", valeur_venale=45000, valeur_a_neuf=52000,
            date_premiere_mise_en_circulation=date(2999, 1, 1),  # in the future
            puissance=6, nombre_place=5, classe=3,
        )


async def test_llm_extraction_is_merged_with_regex():
    llm = FakeListChatModel(responses=['{"valeur_venale": 30000, "puissance": null, "classe": 4}'])
    info = await extract_vehicle_info("CIN 12345678, 7 CV, valeur 30 mille dinars", llm)
    assert info.n_cin == "12345678"  # from the regex
    assert info.puissance == 7  # LLM said null: the regex value is kept
    assert info.valeur_venale == 30000  # from the LLM
    assert info.classe == 4


async def test_llm_garbage_falls_back_to_regex():
    llm = FakeListChatModel(responses=["désolé, je ne sais pas"])
    info = await extract_vehicle_info("CIN 12345678", llm)
    assert info.n_cin == "12345678"
