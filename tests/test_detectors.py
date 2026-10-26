import pytest

from src.core.detectors import (
    PersonalDataDetector,
    QuotationDetector,
    parse_identity,
    parse_phone,
    parse_verification_code,
)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Je veux voir mes contrats", True),
        ("Montrez-moi mes DONNÉES", True),
        ("Quelle est ma prime annuelle ?", True),
        ("Qu'est-ce que la garantie bris de glace ?", False),
        ("Comment déclarer un sinistre ?", False),
    ],
)
def test_personal_data_detection(question, expected):
    assert PersonalDataDetector().needs_personal_data(question) is expected


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Je veux un devis pour ma voiture", True),
        ("Combien coûte une assurance auto ?", True),
        ("Quel est le prix pour un véhicule de 6 CV ?", True),
        ("Qu'est-ce que la prime d'assurance ?", False),  # definition, not a quote request
        ("Combien de garanties couvre le contrat habitation ?", False),  # 'combien' without a vehicle
        ("C'est quoi un devis ?", False),
    ],
)
def test_quotation_detection(question, expected):
    assert QuotationDetector().needs_quotation(question) is expected


def test_product_type():
    detector = QuotationDetector()
    assert detector.product_type("devis pour ma voiture") == "auto"
    assert detector.product_type("devis assurance habitation") == "habitation"
    assert detector.product_type("je veux un devis") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("98 765 432", "98765432"),
        ("+216 98-765-432", "98765432"),
        ("00216 98765432", "98765432"),
        ("12345", None),
    ],
)
def test_parse_phone(text, expected):
    assert parse_phone(text) == expected


def test_parse_identity_short_form():
    identity = parse_identity("Mariem Ben Ali 98 765 432")
    assert (identity.prenom, identity.nom, identity.telephone) == ("Mariem", "Ben Ali", "98765432")
    assert identity.complete


def test_parse_identity_labelled_form():
    identity = parse_identity("nom: Ben Ali, prénom: Mariem, tél: 98765432")
    assert (identity.prenom, identity.nom, identity.telephone) == ("Mariem", "Ben Ali", "98765432")


def test_parse_identity_sentence():
    identity = parse_identity("Je m'appelle Mariem Ben Ali, mon numéro est 98 765 432")
    assert (identity.prenom, identity.nom, identity.telephone) == ("Mariem", "Ben Ali", "98765432")


def test_parse_identity_ignores_the_request_words():
    identity = parse_identity("je veux mes contrats")
    assert not identity.complete


def test_parse_verification_code():
    assert parse_verification_code("mon code est 042917") == "042917"
    assert parse_verification_code("98765432") is None  # 8 digits is not a code
