"""Shared names between the ingestion scripts and the retrievers.

The split between public and private data is the core security rule of the project:
retrievers that answer anonymous questions may only read the PUBLIC_* collections.
Client records (people, contracts, claims, quotes) live in PostgreSQL and are read
only after the client has authenticated, filtered on their own id.
"""

# Qdrant: payload["source_type"] values that anonymous questions may retrieve.
PUBLIC_VECTOR_TYPES = ("document", "garantie", "produit", "faq", "dictionnaire")

# Neo4j: node labels that anonymous questions may retrieve. Every public node has
# a `nom` (short title) and a `texte` (full description) property.
PUBLIC_GRAPH_LABELS = ("Branche", "SousBranche", "Produit", "Garantie", "ProfilCible", "QuestionReponse")

# Full-text index over PUBLIC_GRAPH_LABELS (created by ingestion/build_graph.py).
GRAPH_FULLTEXT_INDEX = "public_knowledge"
