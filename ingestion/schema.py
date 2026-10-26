"""Where each piece of information lives in the hackathon files.

load_tables.py turns every sheet into a table named <file>_<sheet> in snake_case,
with snake_case column names. The names below come from the hackathon version of
the project. If a file or column is named differently, add the new name to the
candidate list: `python -m ingestion.check_data` shows what was found.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A logical table: the first existing table whose name ends with one of `tables`."""

    tables: tuple[str, ...]
    columns: dict[str, tuple[str, ...]]  # logical column -> candidate column names
    required: tuple[str, ...] = ()
    description: str = ""


# --- Private client records (read only after authentication) --------------------------

CLIENTS = Source(
    tables=("donnees_assurance_s1_clients", "s1_clients", "clients_sheet1"),
    columns={
        "id": ("id", "id_client", "client_id", "code_client"),
        "nom": ("nom", "nom_client"),
        "prenom": ("prenom", "prenom_client"),
        "telephone": ("telephone", "tel", "num_tel", "numero_telephone", "gsm"),
        "email": ("email", "mail", "e_mail", "adresse_email"),
    },
    required=("id", "nom", "prenom", "telephone", "email"),
    description="clients who can authenticate (name + phone, code sent to their e-mail)",
)

CLIENT_QUOTES = Source(
    tables=("donnees_assurance_s1_devis", "s1_devis"),
    columns={
        "client_id": ("client_id", "id_client", "code_client"),
        "date": ("created_at", "date_devis", "date_creation", "date"),
    },
    required=("client_id",),
    description="quotes of each client",
)

CLIENT_CONTRACTS = Source(
    tables=("donnees_assurance_s1_contrats", "s1_contrats"),
    columns={
        "client_id": ("client_id", "id_client", "code_client"),
        "date": ("dateeffet", "date_effet", "date_debut", "date_souscription"),
    },
    required=("client_id",),
    description="contracts of each client",
)

# --- Public catalogue (knowledge graph and vector index) ------------------------------

PORTFOLIO_CONTRACTS = Source(
    tables=("donnees_assurance_s1_2_s2_2_contrats", "s2_2_contrats"),
    columns={
        "num_contrat": ("num_contrat", "numero_contrat", "id_contrat", "n_contrat"),
        "branche": ("lib_branche", "branche"),
        "sous_branche": ("lib_sous_branche", "sous_branche"),
        "produit": ("lib_produit", "produit", "nom_produit"),
    },
    required=("produit",),
    description="contract portfolio: only distinct product / branch names are used, no client data",
)

CONTRACT_GUARANTEES = Source(
    tables=("donnees_assurance_s1_2_s2_2_garantie_contrat", "garantie_contrat"),
    columns={
        "num_contrat": ("num_contrat", "numero_contrat", "id_contrat", "n_contrat"),
        "garantie": ("lib_garantie", "garantie", "nom_garantie"),
    },
    required=("num_contrat", "garantie"),
    description="which guarantees appear in contracts of each product",
)

GUARANTEE_DESCRIPTIONS = Source(
    tables=("description_garanties_sheet1", "description_garanties"),
    columns={
        "nom": ("nom_garantie", "lib_garantie", "garantie", "libelle"),
        "description": ("description", "description_garantie", "definition"),
        "type": ("type", "type_garantie"),
    },
    required=("nom",),
    description="description of each guarantee",
)

PRODUCT_PROFILES = Source(
    tables=("mapping_produits_vs_profils_cibles_sheet1", "mapping_produits_vs_profils_cibles"),
    columns={
        "produit": ("produit", "lib_produit", "nom_produit"),
        "profil": ("profil", "profil_cible", "profils_cibles", "cible", "segment"),
    },
    required=("produit", "profil"),
    description="target customer profiles of each product",
)

FAQ = Source(
    tables=("exemples_questions_reponses_qr", "exemples_questions_reponses"),
    columns={
        "question": ("question", "questions"),
        "reponse": ("reponse", "reponses", "response", "answer"),
    },
    required=("question", "reponse"),
    description="example questions and answers",
)

ALL_SOURCES = {
    "clients": CLIENTS,
    "client_quotes": CLIENT_QUOTES,
    "client_contracts": CLIENT_CONTRACTS,
    "portfolio_contracts": PORTFOLIO_CONTRACTS,
    "contract_guarantees": CONTRACT_GUARANTEES,
    "guarantee_descriptions": GUARANTEE_DESCRIPTIONS,
    "product_profiles": PRODUCT_PROFILES,
    "faq": FAQ,
}
