#!/usr/bin/env python3
"""
Orchestrateur RAG Hybride avec LangChain
- Combine recherche vectorielle (Qdrant) et recherche graphe (Neo4j)
- Utilise LangChain pour l'orchestration des multi-retrievers
- Intègre la génération de devis via API
- Utilise DeepSeek Chat v3.0 via OpenRouter
"""

import os

from src.config import CONFIG
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import httpx
from dataclasses import dataclass
import re

# Imports LangChain (versions mises à jour)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Imports pour les bases de données
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from neo4j import GraphDatabase
import psycopg2

# Imports pour les retrievers personnalisés
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_rag_langchain.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Résultat de recherche unifié"""
    content: str
    source: str  # 'vector' ou 'graph'
    score: float
    metadata: Dict[str, Any]
    node_id: Optional[str] = None

class QdrantVectorRetriever(BaseRetriever):
    """
    Retriever personnalisé pour Qdrant qui gère correctement le mapping des champs
    
    Données disponibles dans Qdrant:
    - document: 2,214 points (PDFs d'assurance)
    - contrat: 1,825 points (données contrats)
    - garantie: 910 points (garanties)
    - table: 51 points (descriptions de colonnes - dictionnaire de données)
    
    Note: Les données clients sont dans Neo4j, pas dans Qdrant
    """
    
    def __init__(self, qdrant_client, collection_name: str, embeddings, top_k: int = 10):
        super().__init__()
        self._qdrant_client = qdrant_client
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._top_k = top_k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Récupère les documents pertinents depuis Qdrant avec recherche optimisée"""
        try:
            # Générer l'embedding de la requête
            query_embedding = self._embeddings.embed_query(query)
            
            documents = []
            
            # 1. Recherche générale (sans filtre)
            search_results = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=self._top_k // 2  # Moitié pour la recherche générale
            )
            
            for result in search_results:
                payload = result.payload
                text = payload.get('text', '')
                
                if text and text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': 'qdrant_vector',
                            'score': result.score,
                            'filename': payload.get('filename', ''),
                            'branche': payload.get('branche', ''),
                            'chunk_index': payload.get('chunk_index', 0),
                            'source_type': payload.get('source_type', 'vector'),
                            'chunk_id': payload.get('chunk_id', '')
                        }
                    )
                    documents.append(doc)
            
            # 2. Recherche intelligente basée sur le dictionnaire de données (tables)
            query_lower = query.lower()
            
            # D'abord, récupérer les descriptions des colonnes pour comprendre le contexte
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                # Récupérer toutes les descriptions de colonnes
                table_results = self._qdrant_client.search(
                    collection_name=self._collection_name,
                    query_vector=query_embedding,
                    query_filter=Filter(
                        must=[
                            FieldCondition(key="source_type", match=MatchValue(value="table"))
                        ]
                    ),
                    limit=20  # Récupérer plus de descriptions pour mieux comprendre
                )
                
                # Analyser les descriptions pour déterminer les types de données pertinents
                relevant_tables = set()
                relevant_columns = set()
                
                for result in table_results:
                    text = result.payload.get('text', '')
                    if text:
                        # Extraire le nom de la table et de la colonne
                        parts = text.split(' | ')
                        if len(parts) >= 2:
                            table_part = parts[0].replace('Nom_feuille: ', '')
                            column_part = parts[1].replace('Nom_colonne: ', '')
                            relevant_tables.add(table_part.lower())
                            relevant_columns.add(column_part.lower())
                
                # Ajouter les descriptions de colonnes pertinentes
                for result in table_results:
                    payload = result.payload
                    text = payload.get('text', '')
                    
                    if text and text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': 'qdrant_tables',
                                'score': result.score,
                                'filename': payload.get('filename', ''),
                                'branche': payload.get('branche', ''),
                                'source_type': 'table',
                                'chunk_id': payload.get('chunk_id', '')
                            }
                        )
                        documents.append(doc)
                
                # Recherche ciblée basée sur les tables identifiées
                if 'personne' in relevant_tables or any(word in query_lower for word in ['client', 'personne', 'nom', 'prénom', 'email', 'téléphone']):
                    # Recherche dans les données clients (mais elles sont dans Neo4j)
                    logger.info("🔍 Requête liée aux clients - recherche dans Neo4j recommandée")
                
                if 'contrat' in relevant_tables or any(word in query_lower for word in ['contrat', 'prime', 'paiement', 'statut', 'num_contrat']):
                    try:
                        contrat_results = self._qdrant_client.search(
                            collection_name=self._collection_name,
                            query_vector=query_embedding,
                            query_filter=Filter(
                                must=[
                                    FieldCondition(key="source_type", match=MatchValue(value="contrat"))
                                ]
                            ),
                            limit=self._top_k // 3
                        )
                        
                        for result in contrat_results:
                            payload = result.payload
                            text = payload.get('text', '')
                            
                            if text and text.strip():
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        'source': 'qdrant_contrats',
                                        'score': result.score,
                                        'branche': payload.get('branche', ''),
                                        'source_type': 'contrat',
                                        'chunk_id': payload.get('chunk_id', '')
                                    }
                                )
                                documents.append(doc)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur recherche contrats: {e}")
                
                if 'garantie' in relevant_tables or any(word in query_lower for word in ['garantie', 'couverture', 'risque', 'protection', 'lib_garantie']):
                    try:
                        garantie_results = self._qdrant_client.search(
                            collection_name=self._collection_name,
                            query_vector=query_embedding,
                            query_filter=Filter(
                                must=[
                                    FieldCondition(key="source_type", match=MatchValue(value="garantie"))
                                ]
                            ),
                            limit=self._top_k // 3
                        )
                        
                        for result in garantie_results:
                            payload = result.payload
                            text = payload.get('text', '')
                            
                            if text and text.strip():
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        'source': 'qdrant_garanties',
                                        'score': result.score,
                                        'branche': payload.get('branche', ''),
                                        'source_type': 'garantie',
                                        'chunk_id': payload.get('chunk_id', '')
                                    }
                                )
                                documents.append(doc)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur recherche garanties: {e}")
                
                if 'sinistre' in relevant_tables or any(word in query_lower for word in ['sinistre', 'accident', 'dommage', 'indemnisation']):
                    try:
                        # Recherche dans les documents PDF pour les sinistres
                        doc_results = self._qdrant_client.search(
                            collection_name=self._collection_name,
                            query_vector=query_embedding,
                            query_filter=Filter(
                                must=[
                                    FieldCondition(key="source_type", match=MatchValue(value="document"))
                                ]
                            ),
                            limit=self._top_k // 3
                        )
                        
                        for result in doc_results:
                            payload = result.payload
                            text = payload.get('text', '')
                            
                            if text and text.strip():
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        'source': 'qdrant_documents',
                                        'score': result.score,
                                        'filename': payload.get('filename', ''),
                                        'branche': payload.get('branche', ''),
                                        'chunk_index': payload.get('chunk_index', 0),
                                        'source_type': 'document',
                                        'chunk_id': payload.get('chunk_id', '')
                                    }
                                )
                                documents.append(doc)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur recherche documents: {e}")
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur recherche intelligente: {e}")
            
            # Note: Les données clients sont dans Neo4j, pas dans Qdrant
            # Qdrant contient: documents (PDFs), contrats, garanties, tables (descriptions)
            # Neo4j contient: clients, relations, données structurées
            
            # Dédupliquer les documents par chunk_id
            seen_chunks = set()
            unique_documents = []
            for doc in documents:
                chunk_id = doc.metadata.get('chunk_id', '')
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_documents.append(doc)
            
            logger.info(f"🔍 Qdrant retriever: {len(unique_documents)} documents trouvés (dédupliqués)")
            return unique_documents[:self._top_k]
            
        except Exception as e:
            logger.error(f"❌ Erreur Qdrant retriever: {e}")
            return []

def convert_neo4j_objects(obj):
    """Convertit les objets Neo4j en types sérialisables"""
    if hasattr(obj, 'items'):
        return {k: convert_neo4j_objects(v) for k, v in obj.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return [convert_neo4j_objects(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # DateTime objects
        return obj.isoformat()
    else:
        return str(obj)

class Neo4jGraphRetriever(BaseRetriever):
    """
    Retriever personnalisé pour Neo4j
    
    Données disponibles dans Neo4j:
    - Contrat: 92,981 nœuds
    - DonneesAssuranceS1_Contrats: 76,829 nœuds
    - PersonnePhysique: 43,314 nœuds
    - Data_donnees_assurance_s1_clients: 14,400 nœuds (CLIENTS)
    - DescriptionGarantie: 910 nœuds
    - Et autres tables de données...
    
    Note: Les clients sont dans Neo4j, pas dans Qdrant
    """
    
    def __init__(self, neo4j_driver, top_k: int = 10):
        super().__init__()
        self._neo4j_driver = neo4j_driver
        self._top_k = top_k
    
    @property
    def neo4j_driver(self):
        return self._neo4j_driver
    
    @property
    def top_k(self):
        return self._top_k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Récupère les documents pertinents depuis Neo4j"""
        try:
            # Extraire les entités clés de la requête
            entities = self._extract_entities(query)
            
            documents = []
            
            with self.neo4j_driver.session() as session:
                # Recherche par entités
                for entity in entities:
                    # Requête pour trouver les relations
                    cypher_query = """
                    MATCH (n)-[r]-(m)
                    WHERE toLower(n.name) CONTAINS toLower($entity) 
                       OR toLower(n.nom) CONTAINS toLower($entity)
                       OR toLower(n.type) CONTAINS toLower($entity)
                       OR toLower(n.description) CONTAINS toLower($entity)
                       OR toLower(n.document) CONTAINS toLower($entity)
                       OR toLower(n.insurance_type) CONTAINS toLower($entity)
                       OR toLower(n.LIB_GARANTIE) CONTAINS toLower($entity)
                       OR toLower(n.LIB_PRODUIT) CONTAINS toLower($entity)
                       OR toLower(n.LIB_BRANCHE) CONTAINS toLower($entity)
                       OR toLower(n.LIB_SOUS_BRANCHE) CONTAINS toLower($entity)
                       OR toLower(n.NOM_PRENOM) CONTAINS toLower($entity)
                       OR toLower(n.RAISON_SOCIALE) CONTAINS toLower($entity)
                       OR toLower(n.Question) CONTAINS toLower($entity)
                       OR toLower(n.Réponse) CONTAINS toLower($entity)
                       OR toLower(n.Nom_colonne) CONTAINS toLower($entity)
                       OR toLower(n.nom_garantie) CONTAINS toLower($entity)
                    RETURN n, r, m, 
                           COALESCE(n.name, '') + ' ' + COALESCE(n.nom, '') + ' ' + 
                           COALESCE(n.description, '') + ' ' + COALESCE(n.document, '') + ' ' + 
                           COALESCE(n.LIB_GARANTIE, '') + ' ' + COALESCE(n.LIB_PRODUIT, '') + ' ' + 
                           COALESCE(n.Question, '') + ' ' + COALESCE(n.Réponse, '') + ' ' + 
                           type(r) + ' ' + COALESCE(m.name, '') as content
                    LIMIT $limit
                    """
                    
                    result = session.run(cypher_query, {"entity": entity, "limit": self.top_k})
                    
                    for record in result:
                        content = record.get('content', '')
                        if content and content.strip():  # Vérifier que le contenu n'est pas vide
                            try:
                                doc = Document(
                                    page_content=str(content),
                                    metadata={
                                        'source': 'neo4j_graph',
                                        'node': convert_neo4j_objects(dict(record['n'])) if record.get('n') else {},
                                        'relation': convert_neo4j_objects(dict(record['r'])) if record.get('r') else {},
                                        'related_node': convert_neo4j_objects(dict(record['m'])) if record.get('m') else {},
                                        'entity': entity,
                                        'source_type': 'graph'
                                    }
                                )
                                documents.append(doc)
                            except Exception as e:
                                logger.warning(f"⚠️ Erreur création document Neo4j: {e}")
                                continue
                
                # Recherche spécifique pour les clients (Data_donnees_assurance_s1_clients)
                clients_query = """
                MATCH (n:Data_donnees_assurance_s1_clients)
                WHERE toLower(n.nom) CONTAINS toLower($query) 
                   OR toLower(n.prénom) CONTAINS toLower($query)
                   OR toLower(n.email) CONTAINS toLower($query)
                   OR toLower(n.téléphone) CONTAINS toLower($query)
                   OR toLower(n.adresse) CONTAINS toLower($query)
                   OR toLower(n.profession) CONTAINS toLower($query)
                RETURN n, 
                       COALESCE(n.nom, '') + ' ' + COALESCE(n.prénom, '') + ' ' + 
                       COALESCE(n.email, '') + ' ' + COALESCE(n.profession, '') + ' ' + 
                       COALESCE(n.adresse, '') as content
                LIMIT $limit
                """
                
                # Recherche spécifique pour les contrats (DonneesAssuranceS1_Contrats)
                contrats_query = """
                MATCH (n:DonneesAssuranceS1_Contrats)
                WHERE toLower(n.produit) CONTAINS toLower($query) 
                   OR toLower(n.branche) CONTAINS toLower($query)
                   OR toLower(n.statutcontrat) CONTAINS toLower($query)
                   OR toLower(n.statutpaiement) CONTAINS toLower($query)
                RETURN n, 
                       COALESCE(n.produit, '') + ' ' + COALESCE(n.branche, '') + ' ' + 
                       COALESCE(n.statutcontrat, '') + ' ' + COALESCE(n.statutpaiement, '') + ' ' +
                       'Prime: ' + COALESCE(toString(n.primeannuelle), '') as content
                LIMIT $limit
                """
                
                # Recherche spécifique pour les garanties (DescriptionGarantie)
                garanties_query = """
                MATCH (n:DescriptionGarantie)
                WHERE toLower(n.nom_garantie) CONTAINS toLower($query) 
                   OR toLower(n.description) CONTAINS toLower($query)
                   OR toLower(n.type) CONTAINS toLower($query)
                RETURN n, 
                       COALESCE(n.nom_garantie, '') + ' ' + COALESCE(n.description, '') + ' ' + 
                       COALESCE(n.type, '') as content
                LIMIT $limit
                """
                
                # Recherche générale dans le graphe
                general_query = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($query) 
                   OR toLower(n.nom) CONTAINS toLower($query)
                   OR toLower(n.type) CONTAINS toLower($query)
                   OR toLower(n.description) CONTAINS toLower($query)
                   OR toLower(n.document) CONTAINS toLower($query)
                   OR toLower(n.insurance_type) CONTAINS toLower($query)
                   OR toLower(n.LIB_GARANTIE) CONTAINS toLower($query)
                   OR toLower(n.LIB_PRODUIT) CONTAINS toLower($query)
                   OR toLower(n.LIB_BRANCHE) CONTAINS toLower($query)
                   OR toLower(n.LIB_SOUS_BRANCHE) CONTAINS toLower($query)
                   OR toLower(n.NOM_PRENOM) CONTAINS toLower($query)
                   OR toLower(n.RAISON_SOCIALE) CONTAINS toLower($query)
                   OR toLower(n.Question) CONTAINS toLower($query)
                   OR toLower(n.Réponse) CONTAINS toLower($query)
                   OR toLower(n.Nom_colonne) CONTAINS toLower($query)
                   OR toLower(n.nom_garantie) CONTAINS toLower($query)
                RETURN n, 
                       COALESCE(n.name, '') + ' ' + COALESCE(n.nom, '') + ' ' + 
                       COALESCE(n.description, '') + ' ' + COALESCE(n.document, '') + ' ' + 
                       COALESCE(n.LIB_GARANTIE, '') + ' ' + COALESCE(n.LIB_PRODUIT, '') + ' ' + 
                       COALESCE(n.Question, '') + ' ' + COALESCE(n.Réponse, '') as content
                LIMIT $limit
                """
                
                # Exécuter la requête clients
                clients_result = session.run(clients_query, {"query": query, "limit": self.top_k})
                
                for record in clients_result:
                    content = record.get('content', '')
                    if content and content.strip():
                        try:
                            doc = Document(
                                page_content=str(content),
                                metadata={
                                    'source': 'neo4j_clients',
                                    'node': convert_neo4j_objects(dict(record['n'])) if record.get('n') else {},
                                    'query': query,
                                    'source_type': 'clients'
                                }
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"⚠️ Erreur création document clients: {e}")
                            continue
                
                # Exécuter la requête contrats
                contrats_result = session.run(contrats_query, {"query": query, "limit": self.top_k})
                
                for record in contrats_result:
                    content = record.get('content', '')
                    if content and content.strip():
                        try:
                            doc = Document(
                                page_content=str(content),
                                metadata={
                                    'source': 'neo4j_contrats',
                                    'node': convert_neo4j_objects(dict(record['n'])) if record.get('n') else {},
                                    'query': query,
                                    'source_type': 'contrats'
                                }
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"⚠️ Erreur création document contrats: {e}")
                            continue
                
                # Exécuter la requête garanties
                garanties_result = session.run(garanties_query, {"query": query, "limit": self.top_k})
                
                for record in garanties_result:
                    content = record.get('content', '')
                    if content and content.strip():
                        try:
                            doc = Document(
                                page_content=str(content),
                                metadata={
                                    'source': 'neo4j_garanties',
                                    'node': convert_neo4j_objects(dict(record['n'])) if record.get('n') else {},
                                    'query': query,
                                    'source_type': 'garanties'
                                }
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"⚠️ Erreur création document garanties: {e}")
                            continue
                
                # Exécuter la requête générale
                result = session.run(general_query, {"query": query, "limit": self.top_k})
                
                for record in result:
                    content = record.get('content', '')
                    if content and content.strip():  # Vérifier que le contenu n'est pas vide
                        try:
                            doc = Document(
                                page_content=str(content),
                                metadata={
                                    'source': 'neo4j_graph',
                                    'node': convert_neo4j_objects(dict(record['n'])) if record.get('n') else {},
                                    'query': query,
                                    'source_type': 'graph'
                                }
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"⚠️ Erreur création document Neo4j général: {e}")
                            continue
            
            logger.info(f"🌐 Neo4j retriever: {len(documents)} documents trouvés")
            return documents[:self.top_k]
            
        except Exception as e:
            logger.error(f"❌ Erreur Neo4j retriever: {e}")
            return []
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extrait les entités clés de la requête"""
        # Mots-clés d'assurance
        insurance_keywords = [
            'assurance', 'contrat', 'garantie', 'sinistre', 'prime', 'devis',
            'vie', 'auto', 'habitation', 'sante', 'prevoyance', 'epargne',
            'personne', 'physique', 'morale', 'produit', 'branche'
        ]
        
        entities = []
        query_lower = query.lower()
        
        for keyword in insurance_keywords:
            if keyword in query_lower:
                entities.append(keyword)
        
        # Ajouter des mots significatifs
        words = query.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ['avec', 'pour', 'dans', 'sur', 'par']:
                entities.append(word.lower())
        
        return entities[:5]  # Limiter à 5 entités

class HybridRetriever(BaseRetriever):
    """Retriever hybride combinant Qdrant et Neo4j"""
    
    def __init__(self, vector_retriever, graph_retriever, vector_weight: float = 0.6, graph_weight: float = 0.4):
        super().__init__()
        self._vector_retriever = vector_retriever
        self._graph_retriever = graph_retriever
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight
    
    @property
    def vector_retriever(self):
        return self._vector_retriever
    
    @property
    def graph_retriever(self):
        return self._graph_retriever
    
    @property
    def vector_weight(self):
        return self._vector_weight
    
    @property
    def graph_weight(self):
        return self._graph_weight
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Combine les résultats des deux retrievers"""
        try:
            # Récupérer les documents des sources disponibles
            vector_docs = []
            if self.vector_retriever is not None:
                try:
                    raw_docs = self.vector_retriever._get_relevant_documents(query, run_manager=run_manager)
                    # Filtrer les documents avec du contenu valide
                    vector_docs = []
                    for doc in raw_docs:
                        if hasattr(doc, 'page_content') and doc.page_content is not None and str(doc.page_content).strip():
                            # S'assurer que le contenu est valide
                            try:
                                content = str(doc.page_content).strip()
                                if content:
                                    # Créer un nouveau document avec le contenu validé
                                    valid_doc = Document(
                                        page_content=content,
                                        metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                                    )
                                    vector_docs.append(valid_doc)
                            except Exception as e:
                                logger.warning(f"⚠️ Erreur validation document: {e}")
                                continue
                        else:
                            logger.warning(f"⚠️ Document ignoré (contenu vide ou None): {doc}")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur retriever vectoriel: {e}")
                    vector_docs = []
            
            graph_docs = []
            if self.graph_retriever is not None:
                try:
                    graph_docs = self.graph_retriever._get_relevant_documents(query, run_manager=run_manager)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur retriever graphe: {e}")
                    graph_docs = []
            
            # Appliquer les poids
            for doc in vector_docs:
                if hasattr(doc, 'metadata'):
                    doc.metadata['weight'] = self.vector_weight
                    doc.metadata['source_type'] = 'vector'
            
            for doc in graph_docs:
                if hasattr(doc, 'metadata'):
                    doc.metadata['weight'] = self.graph_weight
                    doc.metadata['source_type'] = 'graph'
            
            # Combiner et dédupliquer
            all_docs = vector_docs + graph_docs
            
            if not all_docs:
                logger.warning("⚠️ Aucun document trouvé par les retrievers")
                return []
            
            # Déduplication basique
            seen_content = set()
            unique_docs = []
            
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            # Trier par poids (simplifié)
            unique_docs.sort(key=lambda x: x.metadata.get('weight', 0), reverse=True)
            
            logger.info(f"🔄 Hybrid retriever: {len(unique_docs)} documents uniques")
            return unique_docs[:15]  # Top 15
            
        except Exception as e:
            logger.error(f"❌ Erreur hybrid retriever: {e}")
            return []

class PersonalDataDetector:
    """Détecteur de demandes d'informations personnelles"""
    
    def __init__(self):
        self.personal_keywords = [
            'mes contrats', 'mon contrat', 'mes devis', 'mon devis',
            'mes sinistres', 'mon sinistre', 'mes informations', 'mon historique',
            'mes données', 'mon profil', 'mes polices', 'ma police',
            'mon assurance', 'mes garanties', 'mon compte',
            'je vais avoir mes contrats', 'je veux mes contrats',
            'je vais avoir mes devis', 'je veux mes devis',
            'je vais avoir mes sinistres', 'je veux mes sinistres',
            'avoir mes contrats', 'avoir mes devis', 'avoir mes sinistres'
        ]
        
        self.identity_patterns = [
            r'nom[:\s]*([a-zA-ZÀ-ÿ\s]+)',
            r'prénom[:\s]*([a-zA-ZÀ-ÿ\s]+)',
            r'téléphone[:\s]*([0-9\s\+\-\(\)]+)',
            r'tel[:\s]*([0-9\s\+\-\(\)]+)',
            r'email[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'mail[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ]
    
    def needs_personal_data(self, query: str) -> bool:
        """Détecte si la requête concerne des données personnelles"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.personal_keywords)
    
    def extract_identity_info(self, query: str) -> Dict[str, Optional[str]]:
        """Extrait les informations d'identité de la requête"""
        identity_info = {
            'nom': None,
            'prénom': None,
            'téléphone': None,
            'email': None
        }
        
        # Patterns pour extraire les informations
        patterns = {
            'nom': [
                r'nom[:\s]*est\s+([a-zA-ZÀ-ÿ\s]+?)(?:\s+prénom|\s+téléphone|\s+tel|$)',
                r'nom[:\s]*([a-zA-ZÀ-ÿ\s]+?)(?:\s+prénom|\s+téléphone|\s+tel|$)',
                r'je m\'appelle\s+([a-zA-ZÀ-ÿ\s]+?)(?:\s+prénom|\s+téléphone|\s+tel|$)',
                r'([A-Z][a-zA-ZÀ-ÿ]+)\s+([A-Z][a-zA-ZÀ-ÿ]+)(?:\s+prénom|\s+téléphone|\s+tel|$)',
                # Pattern pour format simple "prénom nom téléphone"
                r'([a-zA-ZÀ-ÿ]+)\s+([a-zA-ZÀ-ÿ]+)\s+([0-9\s\+\-\(\)]+)'
            ],
            'prénom': [
                r'prénom[:\s]*([a-zA-ZÀ-ÿ\s]+?)(?:\s+téléphone|\s+tel|$)',
                r'prénom[:\s]*est\s+([a-zA-ZÀ-ÿ\s]+?)(?:\s+téléphone|\s+tel|$)',
                # Pattern pour format simple "prénom nom téléphone" (premier mot)
                r'^([a-zA-ZÀ-ÿ]+)\s+([a-zA-ZÀ-ÿ]+)\s+([0-9\s\+\-\(\)]+)'
            ],
            'téléphone': [
                r'téléphone[:\s]*([0-9\s\+\-\(\)]+)', 
                r'tel[:\s]*([0-9\s\+\-\(\)]+)',
                # Pattern pour format simple "prénom nom téléphone" (dernier mot)
                r'^([a-zA-ZÀ-ÿ]+)\s+([a-zA-ZÀ-ÿ]+)\s+([0-9\s\+\-\(\)]+)'
            ],
            'email': [r'email[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'mail[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})']
        }
        
        # Gestion spéciale pour le format simple "prénom nom téléphone"
        simple_format_match = re.search(r'^([a-zA-ZÀ-ÿ]+)\s+([a-zA-ZÀ-ÿ]+)\s+([0-9\s\+\-\(\)]+)$', query.strip(), re.IGNORECASE)
        if simple_format_match:
            identity_info['prénom'] = simple_format_match.group(1).strip()
            identity_info['nom'] = simple_format_match.group(2).strip()
            identity_info['téléphone'] = simple_format_match.group(3).strip()
        else:
            # Extraction normale avec patterns
            for field, field_patterns in patterns.items():
                for pattern in field_patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        identity_info[field] = match.group(1).strip()
                        break
        
        return identity_info
    
    def is_verification_code(self, query: str) -> bool:
        """Détecte si la requête contient un code de vérification"""
        # Code à 6 chiffres
        code_pattern = r'\b\d{6}\b'
        return bool(re.search(code_pattern, query))

class QuotationDetector:
    """Détecteur de besoin de devis"""
    
    def __init__(self):
        self.quotation_keywords = [
            'devis', 'prix', 'cout', 'tarif', 'montant', 'prime',
            'combien', 'quel prix', 'estimation', 'calculer'
        ]
    
    def needs_quotation(self, query: str) -> bool:
        """Détecte si la requête nécessite un devis"""
        query_lower = query.lower()
        
        # Mots-clés qui indiquent une question de définition (pas un devis)
        definition_keywords = [
            'qu\'est-ce que', 'qu\'est ce que', 'c\'est quoi', 'définition', 'définir',
            'expliquer', 'explique', 'signifie', 'signification', 'que veut dire',
            'comment ça marche', 'comment fonctionne', 'à quoi sert'
        ]
        
        # Si c'est une question de définition, ne pas déclencher de devis
        if any(keyword in query_lower for keyword in definition_keywords):
            return False
        
        # Mots-clés de devis avec contexte plus spécifique
        quotation_phrases = [
            'je veux un devis', 'j\'ai besoin d\'un devis', 'donnez-moi un devis',
            'calculer le prix', 'estimer le coût', 'combien coûte',
            'quel est le prix de', 'tarif pour', 'devis pour'
        ]
        
        # Vérifier les phrases complètes de devis
        if any(phrase in query_lower for phrase in quotation_phrases):
            return True
        
        # Vérifier les mots-clés simples seulement si ce n'est pas une question de définition
        return any(keyword in query_lower for keyword in self.quotation_keywords)
    
    def extract_product_type(self, query: str) -> str:
        """Extrait le type de produit de la requête"""
        product_types = {
            'auto': ['auto', 'voiture', 'vehicule', 'automobile', 'assurance auto'],
            'vie': ['assurance vie', 'epargne', 'capitalisation'],
            'habitation': ['habitation', 'maison', 'appartement', 'logement'],
            'sante': ['sante', 'medical', 'hospitalisation'],
            'prevoyance': ['prevoyance', 'invalidite', 'deces']
        }
        
        query_lower = query.lower()
        
        for product_type, keywords in product_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return product_type
        
        return 'auto'  # Par défaut pour l'API disponible

class QuotationGenerator:
    """Générateur de devis via API"""
    
    def __init__(self, api_config: Dict[str, str]):
        self.api_config = api_config
    
    def _extract_vehicle_info_from_query(self, query: str) -> Dict[str, Any]:
        """Extrait les informations véhicule de la requête de manière intelligente"""
        import re
        
        # Valeurs par défaut basées sur l'exemple de l'API
        vehicle_info = {
            'n_cin': '00000000',  # CIN par défaut
            'valeur_venale': 60000,  # Valeur vénale
            'nature_contrat': 'r',  # Nature du contrat (r = responsabilité civile)
            'nombre_place': 5,  # Nombre de places
            'valeur_a_neuf': 60000,  # Valeur à neuf
            'date_premiere_mise_en_circulation': '2022-02-28',  # Date de mise en circulation
            'capital_bris_de_glace': 900,  # Capital bris de glace
            'capital_dommage_collision': 60000,  # Capital dommage collision
            'puissance': 6,  # Puissance fiscale
            'classe': 3  # Classe de bonus/malus
        }
        
        query_lower = query.lower()
        
        # Extraction intelligente des informations
        
        # 1. Valeur du véhicule
        valeur_patterns = [
            r'valeur[:\s]*(\d+)',
            r'prix[:\s]*(\d+)',
            r'cout[:\s]*(\d+)',
            r'tarif[:\s]*(\d+)',
            r'montant[:\s]*(\d+)',
            r'(\d+)\s*euros?',
            r'(\d+)\s*€'
        ]
        
        for pattern in valeur_patterns:
            match = re.search(pattern, query_lower)
            if match:
                valeur = int(match.group(1))
                vehicle_info['valeur_venale'] = valeur
                vehicle_info['valeur_a_neuf'] = valeur
                break
        
        # 2. Date de mise en circulation
        date_patterns = [
            r'(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})',
            r'(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})',
            r'(\d{4})',
            r'(\d{1,2})[-\/](\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 3:
                    year, month, day = match.groups()
                    if len(year) == 4:
                        vehicle_info['date_premiere_mise_en_circulation'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    else:
                        vehicle_info['date_premiere_mise_en_circulation'] = f"{day.zfill(2)}-{month.zfill(2)}-{year}"
                elif len(match.groups()) == 1:
                    year = match.group(1)
                    if len(year) == 4:
                        vehicle_info['date_premiere_mise_en_circulation'] = f"{year}-01-01"
                break
        
        # 3. Puissance fiscale
        puissance_patterns = [
            r'puissance[:\s]*(\d+)',
            r'(\d+)\s*cv',
            r'(\d+)\s*chevaux',
            r'(\d+)\s*fiscaux?'
        ]
        
        for pattern in puissance_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_info['puissance'] = int(match.group(1))
                break
        
        # 4. Classe bonus/malus
        classe_patterns = [
            r'classe[:\s]*(\d+)',
            r'bonus[:\s]*(\d+)',
            r'malus[:\s]*(\d+)',
            r'(\d+)\s*classe'
        ]
        
        for pattern in classe_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_info['classe'] = int(match.group(1))
                break
        
        # 5. Nombre de places
        places_patterns = [
            r'(\d+)\s*places?',
            r'(\d+)\s*personnes?',
            r'(\d+)\s*passagers?',
            r'(\d+)\s*sièges?'
        ]
        
        for pattern in places_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_info['nombre_place'] = int(match.group(1))
                break
        
        # 6. Type de contrat (API n'accepte que 'r' et 'n')
        if 'tous risques' in query_lower or 'tous-risques' in query_lower:
            vehicle_info['nature_contrat'] = 'n'  # Changé de 'a' à 'n'
        elif 'collision' in query_lower:
            vehicle_info['nature_contrat'] = 'r'  # Changé de 'c' à 'r'
        elif 'tiers' in query_lower:
            vehicle_info['nature_contrat'] = 'r'  # Changé de 't' à 'r'
        # 'r' reste par défaut pour responsabilité civile
        
        # 7. CIN (si mentionné)
        cin_patterns = [
            r'cin[:\s]*(\d{8})',
            r'carte[:\s]*d\'identité[:\s]*(\d{8})',
            r'(\d{8})'
        ]
        
        for pattern in cin_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_info['n_cin'] = match.group(1)
                break
        
        # 8. Capital bris de glace
        bris_patterns = [
            r'bris[:\s]*de[:\s]*glace[:\s]*(\d+)',
            r'glace[:\s]*(\d+)',
            r'(\d+)\s*bris'
        ]
        
        for pattern in bris_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_info['capital_bris_de_glace'] = int(match.group(1))
                break
        
        # 9. Capital dommage collision
        collision_patterns = [
            r'dommage[:\s]*collision[:\s]*(\d+)',
            r'collision[:\s]*(\d+)',
            r'(\d+)\s*collision'
        ]
        
        for pattern in collision_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_info['capital_dommage_collision'] = int(match.group(1))
                break
        
        return vehicle_info
    
    def _validate_vehicle_info(self, vehicle_info: Dict[str, Any]) -> Dict[str, Any]:
        """Valide et corrige les informations véhicule"""
        errors = []
        warnings = []
        
        # Validation CIN
        if not vehicle_info['n_cin'] or len(vehicle_info['n_cin']) != 8:
            errors.append("CIN invalide (doit contenir 8 chiffres)")
        
        # Validation valeur vénale
        if vehicle_info['valeur_venale'] < 1000:
            warnings.append("Valeur vénale très faible")
        elif vehicle_info['valeur_venale'] > 500000:
            warnings.append("Valeur vénale très élevée")
        
        # Validation date
        try:
            from datetime import datetime
            datetime.strptime(vehicle_info['date_premiere_mise_en_circulation'], '%Y-%m-%d')
        except ValueError:
            errors.append("Format de date invalide")
        
        # Validation puissance
        if vehicle_info['puissance'] < 1 or vehicle_info['puissance'] > 20:
            warnings.append("Puissance fiscale inhabituelle")
        
        # Validation classe
        if vehicle_info['classe'] < 1 or vehicle_info['classe'] > 50:
            errors.append("Classe bonus/malus invalide")
        
        # Validation nombre de places
        if vehicle_info['nombre_place'] < 2 or vehicle_info['nombre_place'] > 9:
            warnings.append("Nombre de places inhabituel")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def generate_quotation(self, query: str, product_type: str) -> Optional[Dict]:
        """Génère un devis via API avec validation et gestion d'erreurs améliorée"""
        try:
            if product_type != 'auto':
                logger.warning(f"⚠️ API disponible uniquement pour l'assurance auto, type demandé: {product_type}")
                return {
                    'error': True,
                    'message': f"Désolé, seuls les devis d'assurance auto sont disponibles pour le moment. Vous avez demandé un devis pour: {product_type}",
                    'available_products': ['auto'],
                    'suggestion': "Essayez une requête comme: 'Je veux un devis pour une assurance auto'"
                }
            
            # Extraire les informations véhicule
            vehicle_info = self._extract_vehicle_info_from_query(query)
            
            # Valider les informations
            validation = self._validate_vehicle_info(vehicle_info)
            
            if not validation['valid']:
                return {
                    'error': True,
                    'message': "Informations véhicule invalides",
                    'errors': validation['errors'],
                    'warnings': validation['warnings'],
                    'extracted_info': vehicle_info
                }
            
            # Construire l'URL avec les paramètres
            base_url = "https://apidevis.onrender.com"
            endpoint = "/api/auto/packs"
            
            # Construire les paramètres de requête
            params = {
                'n_cin': vehicle_info['n_cin'],
                'valeur_venale': vehicle_info['valeur_venale'],
                'nature_contrat': vehicle_info['nature_contrat'],
                'nombre_place': vehicle_info['nombre_place'],
                'valeur_a_neuf': vehicle_info['valeur_a_neuf'],
                'date_premiere_mise_en_circulation': vehicle_info['date_premiere_mise_en_circulation'],
                'capital_bris_de_glace': vehicle_info['capital_bris_de_glace'],
                'capital_dommage_collision': vehicle_info['capital_dommage_collision'],
                'puissance': vehicle_info['puissance'],
                'classe': vehicle_info['classe']
            }
            
            logger.info(f"🚗 Appel API devis auto avec paramètres: {params}")
            
            # Appel à l'API de devis (GET avec paramètres) - async
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}{endpoint}",
                    params=params,
                    timeout=30
                )
            
            if response.status_code == 200:
                quotation_result = response.json()
                logger.info("✅ Devis auto généré avec succès")
                
                # Enrichir le résultat avec les paramètres utilisés
                quotation_result['_metadata'] = {
                    'api_used': 'apidevis.onrender.com',
                    'parameters_used': params,
                    'query_original': query,
                    'timestamp': datetime.now().isoformat(),
                    'warnings': validation.get('warnings', [])
                }
                
                return quotation_result
            else:
                # Gestion d'erreurs détaillée
                error_message = f"Erreur API devis: {response.status_code}"
                if response.status_code == 400:
                    error_message += " - Paramètres invalides"
                elif response.status_code == 404:
                    error_message += " - Endpoint non trouvé"
                elif response.status_code == 500:
                    error_message += " - Erreur serveur interne"
                elif response.status_code == 503:
                    error_message += " - Service temporairement indisponible"
                
                logger.error(f"❌ {error_message} - {response.text}")
                
                return {
                    'error': True,
                    'message': error_message,
                    'status_code': response.status_code,
                    'response_text': response.text,
                    'parameters_used': params
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur génération devis: {e}")
            return None

class HybridRAGLangChainOrchestrator:
    """Orchestrateur principal du RAG hybride avec LangChain"""
    
    def __init__(self):
        self.vector_retriever = None
        self.graph_retriever = None
        self.hybrid_retriever = None
        self.llm = None
        self.memory = None
        self.quotation_detector = None
        self.quotation_generator = None
        self.personal_data_detector = None
        self.auth_system = None
        self.pending_auth = {}  # {user_id: {step, data}}
        
        # Configuration (read from environment variables, see .env.example)
        self.config = CONFIG

    async def initialize(self):
        """Initialise toutes les connexions et composants LangChain"""
        try:
            # Connexion Qdrant
            qdrant_client = QdrantClient(
                host=self.config['qdrant']['host'],
                port=self.config['qdrant']['port']
            )
            logger.info("✅ Connexion Qdrant établie")
            
            # Connexion Neo4j
            neo4j_driver = GraphDatabase.driver(
                self.config['neo4j']['uri'],
                auth=self.config['neo4j']['auth']
            )
            logger.info("✅ Connexion Neo4j établie")
            
            # Connexion PostgreSQL
            postgres_connection = psycopg2.connect(
                **self.config['postgres']
            )
            logger.info("✅ Connexion PostgreSQL établie")
            
            # Modèle d'embeddings
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2',
                model_kwargs={'device': device}
            )
            logger.info("✅ Modèle d'embeddings chargé")
            
            # Vérifier et créer la collection Qdrant si nécessaire
            try:
                collections = qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if self.config['qdrant']['collection_name'] not in collection_names:
                    logger.warning(f"⚠️ Collection {self.config['qdrant']['collection_name']} non trouvée")
                    logger.info("💡 Assurez-vous que les embeddings sont stockés dans Qdrant")
                    # Créer un retriever vide pour éviter les erreurs
                    self.vector_retriever = None
                else:
                    # Créer le retriever vectoriel personnalisé avec Qdrant
                    self.vector_retriever = QdrantVectorRetriever(
                        qdrant_client=qdrant_client,
                        collection_name=self.config['qdrant']['collection_name'],
                        embeddings=embeddings,
                        top_k=10
                    )
                    logger.info("✅ Retriever vectoriel configuré")
                    
            except Exception as e:
                logger.error(f"❌ Erreur configuration Qdrant: {e}")
                self.vector_retriever = None
            
            # Créer le retriever graphe avec Neo4j
            self.graph_retriever = Neo4jGraphRetriever(
                neo4j_driver=neo4j_driver,
                top_k=10
            )
            logger.info("✅ Retriever graphe configuré")
            
            # Créer le retriever hybride (ou utiliser seulement le graphe si pas de vector)
            if self.vector_retriever is not None:
                self.hybrid_retriever = HybridRetriever(
                    vector_retriever=self.vector_retriever,
                    graph_retriever=self.graph_retriever,
                    vector_weight=0.6,
                    graph_weight=0.4
                )
                logger.info("✅ Retriever hybride configuré")
            else:
                # Utiliser seulement le retriever graphe
                self.hybrid_retriever = self.graph_retriever
                logger.info("✅ Retriever graphe uniquement (pas de vector)")
            
            # Test des retrievers supprimé pour éviter les requêtes automatiques
            
            # Configurer le LLM avec OpenRouter
            if self.config['openrouter']['api_key']:
                self.llm = ChatOpenAI(
                    model=self.config['openrouter']['model'],  # Utiliser 'model' au lieu de 'model_name'
                    openai_api_key=self.config['openrouter']['api_key'],
                    openai_api_base=self.config['openrouter']['base_url'],
                    temperature=0.7,
                    # max_tokens supprimé pour permettre des réponses plus longues
                    request_timeout=30,  # Timeout de 30 secondes
                    max_retries=3,  # Maximum 3 tentatives
                    default_headers={
                        "HTTP-Referer": self.config['openrouter']['http_referer'],
                        "X-Title": "RAG Hybride Assurance"
                    }
                )
                logger.info("✅ LLM OpenRouter configuré")
            else:
                logger.warning("⚠️ Clé API OpenRouter manquante")
            
            # Configurer la mémoire
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info("✅ Mémoire configurée")
            
            # Configurer les composants de devis
            self.quotation_detector = QuotationDetector()
            self.quotation_generator = QuotationGenerator(self.config['quotation_api'])
            self.personal_data_detector = PersonalDataDetector()
            
            # Initialiser le système d'authentification (vraie base de données)
            from real_database_auth import real_auth
            self.auth_system = real_auth
            logger.info("✅ Composants de devis configurés")
                
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            raise
    
    def create_rag_chain(self):
        """Crée la chaîne RAG avec LangChain"""
        try:
            # Template de prompt pour le contexte
            contextualize_q_system_prompt = """Étant donné un historique de conversation et la dernière question de l'utilisateur qui pourrait faire référence au contexte de l'historique, formulez une question autonome qui peut être comprise sans l'historique de conversation. Ne répondez PAS à la question, reformulez-la simplement si nécessaire et retournez-la telle quelle."""
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Créer le retriever avec historique
            history_aware_retriever = create_history_aware_retriever(
                self.llm, 
                self.hybrid_retriever, 
                contextualize_q_prompt
            )
            
            # Template de prompt pour la réponse
            qa_system_prompt = """Vous êtes un expert en assurance français. Répondez à la question de l'utilisateur en utilisant le contexte fourni.

CONTEXTE:
{context}

INSTRUCTIONS:
1. Répondez de manière claire et précise
2. Utilisez le contexte pour enrichir votre réponse
3. Mentionnez les sources (vectoriel/graphe) quand pertinent
4. Répondez en français
5. Si un devis est fourni, expliquez-le clairement

QUESTION: {input}

RÉPONSE:"""
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Créer la chaîne de génération de documents
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            
            # Créer la chaîne RAG finale
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            logger.info("✅ Chaîne RAG créée avec succès")
            return rag_chain
            
        except Exception as e:
            logger.error(f"❌ Erreur création chaîne RAG: {e}")
            raise
    
    async def process_query(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """Traite une requête complète avec LangChain"""
        try:
            logger.info(f"🔍 Traitement de la requête: {query}")
            
            # Vérifier si l'utilisateur est en cours d'authentification
            if user_id in self.pending_auth:
                return await self._handle_auth_step(query, user_id)
            
            # Détection de demande de données personnelles
            if self.personal_data_detector.needs_personal_data(query):
                return await self._handle_personal_data_request(query, user_id)
            
            # Détection de besoin de devis
            needs_quotation = self.quotation_detector.needs_quotation(query)
            quotation = None
            
            if needs_quotation:
                product_type = self.quotation_detector.extract_product_type(query)
                quotation = await self.quotation_generator.generate_quotation(query, product_type)
            
            # Créer la chaîne RAG
            rag_chain = self.create_rag_chain()
            
            # Préparer le contexte avec le devis si disponible
            input_data = {
                "input": query,
                "chat_history": self.memory.chat_memory.messages
            }
            
            if quotation:
                if quotation.get('error'):
                    # Message d'erreur pour devis non disponible
                    input_data["input"] += f"\n\nINFORMATION DEVIS: {quotation.get('message', 'Devis non disponible')}"
                    if quotation.get('suggestion'):
                        input_data["input"] += f"\n\nSUGGESTION: {quotation.get('suggestion')}"
                else:
                    # Devis valide
                    input_data["input"] += f"\n\nDEVIS DISPONIBLE: {json.dumps(quotation, indent=2)}"
            
            # Exécuter la chaîne RAG de manière asynchrone
            result = await rag_chain.ainvoke(input_data)
            
            # Debug: Afficher le résultat complet
            logger.info(f"🔍 Résultat RAG: {result}")
            
            # Sauvegarder dans la mémoire
            self.memory.save_context(
                {"input": query},
                {"output": result["answer"]}
            )
            
            # Compter les sources (gestion des différents formats de retour LangChain)
            context_docs = result.get("context", [])
            if not context_docs and "source_documents" in result:
                context_docs = result["source_documents"]
            
            logger.info(f"📚 Documents récupérés: {len(context_docs)}")
            for i, doc in enumerate(context_docs[:3]):  # Afficher les 3 premiers
                logger.info(f"📄 Doc {i}: {doc.page_content[:100]}... (metadata: {doc.metadata})")
            
            vector_count = sum(1 for doc in context_docs if hasattr(doc, 'metadata') and doc.metadata.get("source_type") == "vector")
            graph_count = sum(1 for doc in context_docs if hasattr(doc, 'metadata') and doc.metadata.get("source_type") == "graph")
            
            return {
                'query': query,
                'response': result["answer"],
                'vector_results_count': vector_count,
                'graph_results_count': graph_count,
                'fused_results_count': len(context_docs),
                'quotation': quotation,
                'needs_quotation': needs_quotation,
                'timestamp': datetime.now().isoformat(),
                'sources': [doc.metadata for doc in context_docs if hasattr(doc, 'metadata')]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement requête: {e}")
            return {
                'query': query,
                'response': f"Erreur: {e}",
                'vector_results_count': 0,
                'graph_results_count': 0,
                'fused_results_count': 0,
                'needs_quotation': False,
                'quotation': None,
                'error': True,
                'timestamp': datetime.now().isoformat(),
                'processing_time': None,
                'sources': None
            }
    
    def clear_memory(self):
        """Efface la mémoire de conversation"""
        self.memory.clear()
        logger.info("🧹 Mémoire effacée")
    
    async def _get_client_email_from_db(self, nom: str, prénom: str, téléphone: str) -> Optional[str]:
        """Récupère l'email du client depuis la base de données (simulateur)"""
        try:
            # Utiliser le simulateur d'authentification
            client = self.auth_system.find_client(nom, prénom, téléphone)
            
            if client:
                email = client['email']
                logger.info(f"✅ Email trouvé pour {prénom} {nom}: {email}")
                return email
            else:
                logger.warning(f"⚠️ Client non trouvé: {prénom} {nom} - {téléphone}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération email: {e}")
            return None
    
    async def _handle_personal_data_request(self, query: str, user_id: str) -> Dict[str, Any]:
        """Gère les demandes de données personnelles"""
        try:
            # Extraire les informations d'identité de la requête
            identity_info = self.personal_data_detector.extract_identity_info(query)
            
            # Vérifier si on a déjà les informations nécessaires
            if identity_info['nom'] and identity_info['prénom'] and identity_info['téléphone']:
                # Rechercher l'email dans la base de données
                email = await self._get_client_email_from_db(
                    identity_info['nom'],
                    identity_info['prénom'],
                    identity_info['téléphone']
                )
                
                if email:
                    # Envoyer le code de vérification directement
                    result = self.auth_system.request_client_verification(
                        identity_info['nom'],
                        identity_info['prénom'],
                        identity_info['téléphone'],
                        email
                    )
                    
                    if result['success']:
                        # Passer directement à l'étape de vérification du code
                        self.pending_auth[user_id] = {
                            'step': 'code',
                            'data': {**identity_info, 'email': email},
                            'original_query': query
                        }
                        return {
                            'query': query,
                            'response': f"Merci {identity_info['prénom']} {identity_info['nom']} ! Code de vérification envoyé à {email}. Veuillez entrer le code reçu par email.",
                            'vector_results_count': 0,
                            'graph_results_count': 0,
                            'fused_results_count': 0,
                            'needs_quotation': False,
                            'quotation': None,
                            'needs_auth': True,
                            'auth_step': 'code',
                            'timestamp': datetime.now().isoformat(),
                            'processing_time': None,
                            'sources': None
                        }
                    else:
                        return {
                            'query': query,
                            'response': f"Erreur lors de l'envoi du code: {result['message']}. Veuillez vérifier vos informations et réessayer.",
                            'vector_results_count': 0,
                            'graph_results_count': 0,
                            'fused_results_count': 0,
                            'needs_quotation': False,
                            'quotation': None,
                            'needs_auth': True,
                            'auth_step': 'identity',
                            'timestamp': datetime.now().isoformat(),
                            'processing_time': None,
                            'sources': None
                        }
                else:
                    return {
                        'query': query,
                        'response': "Client non trouvé dans notre base de données. Veuillez vérifier vos informations (nom, prénom, téléphone) et réessayer.",
                        'vector_results_count': 0,
                        'graph_results_count': 0,
                        'fused_results_count': 0,
                        'needs_quotation': False,
                        'quotation': None,
                        'needs_auth': True,
                        'auth_step': 'identity',
                        'timestamp': datetime.now().isoformat(),
                        'processing_time': None,
                        'sources': None
                    }
            else:
                # Demander les informations d'identité
                self.pending_auth[user_id] = {
                    'step': 'identity',
                    'data': {},
                    'original_query': query
                }
                return {
                    'query': query,
                    'response': "Pour accéder à vos informations personnelles (contrats, devis, sinistres), j'ai besoin de vérifier votre identité. Veuillez me donner votre nom, prénom et numéro de téléphone.",
                    'vector_results_count': 0,
                    'graph_results_count': 0,
                    'fused_results_count': 0,
                    'needs_quotation': False,
                    'quotation': None,
                    'needs_auth': True,
                    'auth_step': 'identity',
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': None,
                    'sources': None
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur gestion données personnelles: {e}")
            return {
                'query': query,
                'response': "Erreur lors de la gestion de votre demande. Veuillez réessayer.",
                'vector_results_count': 0,
                'graph_results_count': 0,
                'fused_results_count': 0,
                'needs_quotation': False,
                'quotation': None,
                'timestamp': datetime.now().isoformat(),
                'processing_time': None,
                'sources': None
            }
    
    async def _handle_auth_step(self, query: str, user_id: str) -> Dict[str, Any]:
        """Gère les étapes d'authentification"""
        try:
            auth_data = self.pending_auth[user_id]
            step = auth_data['step']
            
            if step == 'identity':
                # Extraire les informations d'identité
                identity_info = self.personal_data_detector.extract_identity_info(query)
                
                if identity_info['nom'] and identity_info['prénom'] and identity_info['téléphone']:
                    # Rechercher l'email dans la base de données
                    email = await self._get_client_email_from_db(
                        identity_info['nom'],
                        identity_info['prénom'],
                        identity_info['téléphone']
                    )
                    
                    if email:
                        # Envoyer le code de vérification directement
                        result = self.auth_system.request_client_verification(
                            identity_info['nom'],
                            identity_info['prénom'],
                            identity_info['téléphone'],
                            email
                        )
                        
                        if result['success']:
                            # Passer directement à l'étape de vérification du code
                            auth_data['step'] = 'code'
                            auth_data['data'] = {**identity_info, 'email': email}
                            return {
                                'query': query,
                                'response': f"Merci {identity_info['prénom']} {identity_info['nom']} ! Code de vérification envoyé à {email}. Veuillez entrer le code reçu par email.",
                                'vector_results_count': 0,
                                'graph_results_count': 0,
                                'fused_results_count': 0,
                                'needs_quotation': False,
                                'quotation': None,
                                'needs_auth': True,
                                'auth_step': 'code',
                                'timestamp': datetime.now().isoformat(),
                                'processing_time': None,
                                'sources': None
                            }
                        else:
                            return {
                                'query': query,
                                'response': f"Erreur lors de l'envoi du code: {result['message']}. Veuillez vérifier vos informations et réessayer.",
                                'vector_results_count': 0,
                                'graph_results_count': 0,
                                'fused_results_count': 0,
                                'needs_quotation': False,
                                'quotation': None,
                                'needs_auth': True,
                                'auth_step': 'identity',
                                'timestamp': datetime.now().isoformat(),
                                'processing_time': None,
                                'sources': None
                            }
                    else:
                        return {
                            'query': query,
                            'response': "Client non trouvé dans notre base de données. Veuillez vérifier vos informations (nom, prénom, téléphone) et réessayer.",
                            'vector_results_count': 0,
                            'graph_results_count': 0,
                            'fused_results_count': 0,
                            'needs_quotation': False,
                            'quotation': None,
                            'needs_auth': True,
                            'auth_step': 'identity',
                            'timestamp': datetime.now().isoformat(),
                            'processing_time': None,
                            'sources': None
                        }
                else:
                    return {
                        'query': query,
                        'response': "Je n'ai pas pu extraire toutes les informations nécessaires. Veuillez me donner votre nom, prénom et numéro de téléphone de manière claire.",
                        'vector_results_count': 0,
                        'graph_results_count': 0,
                        'fused_results_count': 0,
                        'needs_quotation': False,
                        'quotation': None,
                        'needs_auth': True,
                        'auth_step': 'identity',
                        'timestamp': datetime.now().isoformat(),
                        'processing_time': None,
                        'sources': None
                    }
            
            elif step == 'code':
                # Vérifier le code
                if self.personal_data_detector.is_verification_code(query):
                    result = self.auth_system.verify_code(
                        auth_data['data']['email'],
                        query.strip()
                    )
                    
                    if result['success']:
                        # Authentification réussie - récupérer les données
                        client_data = self.auth_system.get_client_data(result['session_token'])
                        
                        if client_data['success']:
                            # Nettoyer l'authentification en cours
                            del self.pending_auth[user_id]
                            
                            # Générer la réponse avec les données
                            response = self._format_client_data_response(client_data, auth_data['original_query'])
                            
                            return {
                                'query': query,
                                'response': response,
                                'vector_results_count': 0,
                                'graph_results_count': 0,
                                'fused_results_count': 0,
                                'needs_quotation': False,
                                'quotation': None,
                                'needs_auth': False,
                                'client_data': client_data,
                                'timestamp': datetime.now().isoformat(),
                                'processing_time': None,
                                'sources': None
                            }
                        else:
                            return {
                                'query': query,
                                'response': f"Erreur lors de la récupération de vos données: {client_data['message']}",
                                'vector_results_count': 0,
                                'graph_results_count': 0,
                                'fused_results_count': 0,
                                'needs_quotation': False,
                                'quotation': None,
                                'needs_auth': True,
                                'auth_step': 'code',
                                'timestamp': datetime.now().isoformat(),
                                'processing_time': None,
                                'sources': None
                            }
                    else:
                        return {
                            'query': query,
                            'response': f"Code incorrect: {result['message']}. Veuillez réessayer.",
                            'vector_results_count': 0,
                            'graph_results_count': 0,
                            'fused_results_count': 0,
                            'needs_quotation': False,
                            'quotation': None,
                            'needs_auth': True,
                            'auth_step': 'code',
                            'timestamp': datetime.now().isoformat(),
                            'processing_time': None,
                            'sources': None
                        }
                else:
                    return {
                        'query': query,
                        'response': "Veuillez entrer le code de vérification à 6 chiffres reçu par email.",
                        'vector_results_count': 0,
                        'graph_results_count': 0,
                        'fused_results_count': 0,
                        'needs_quotation': False,
                        'quotation': None,
                        'needs_auth': True,
                        'auth_step': 'code',
                        'timestamp': datetime.now().isoformat(),
                        'processing_time': None,
                        'sources': None
                    }
            
            return {
                'query': query,
                'response': "Erreur dans le processus d'authentification. Veuillez recommencer.",
                'vector_results_count': 0,
                'graph_results_count': 0,
                'fused_results_count': 0,
                'needs_quotation': False,
                'quotation': None,
                'timestamp': datetime.now().isoformat(),
                'processing_time': None,
                'sources': None
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur étape authentification: {e}")
            return {
                'query': query,
                'response': "Erreur lors de l'authentification. Veuillez recommencer.",
                'vector_results_count': 0,
                'graph_results_count': 0,
                'fused_results_count': 0,
                'needs_quotation': False,
                'quotation': None,
                'timestamp': datetime.now().isoformat(),
                'processing_time': None,
                'sources': None
            }
    
    def _format_client_data_response(self, client_data: Dict[str, Any], original_query: str) -> str:
        """Formate la réponse avec les données client"""
        try:
            response = f"🔐 **Authentification réussie !**\n\n"
            
            # Informations personnelles
            client_info = client_data['client_info']
            response += f"**👤 Informations personnelles :**\n"
            response += f"- Nom: {client_info.get('nom', 'N/A')}\n"
            response += f"- Prénom: {client_info.get('prénom', 'N/A')}\n"
            response += f"- Téléphone: {client_info.get('téléphone', 'N/A')}\n"
            response += f"- Email: {client_info.get('email', 'N/A')}\n\n"
            
            # Résumé
            summary = client_data['summary']
            response += f"**📊 Résumé :**\n"
            response += f"- Nombre de devis: {summary['total_devis']}\n"
            response += f"- Nombre de contrats: {summary['total_contrats']}\n\n"
            
            # Devis
            if client_data['devis']:
                response += f"**💰 Vos devis récents :**\n"
                for i, devis in enumerate(client_data['devis'][:3], 1):
                    response += f"{i}. Devis #{devis.get('num_devis', 'N/A')}\n"
                    response += f"   - Produit: {devis.get('produit', 'N/A')}\n"
                    response += f"   - Montant: {devis.get('montant', 0):.2f} DT\n"
                    response += f"   - Date: {devis.get('date', 'N/A')}\n\n"
            
            # Contrats
            if client_data['contrats']:
                response += f"**📋 Vos contrats :**\n"
                for i, contrat in enumerate(client_data['contrats'][:3], 1):
                    response += f"{i}. Contrat #{contrat.get('num_contrat', 'N/A')}\n"
                    response += f"   - Type: {contrat.get('type', 'N/A')}\n"
                    response += f"   - Branche: {contrat.get('branche', 'N/A')}\n"
                    response += f"   - Statut: {contrat.get('statut', 'N/A')}\n"
                    response += f"   - Montant: {contrat.get('montant', 0):.2f} DT\n"
                    response += f"   - Capital assuré: {contrat.get('capital_assure', 0):.2f} DT\n"
                    response += f"   - Paiement: {contrat.get('statut_paiement', 'N/A')}\n"
                    response += f"   - Période: {contrat.get('date_debut', 'N/A')} → {contrat.get('date_fin', 'N/A')}\n\n"
            
            response += "💡 Vous pouvez me poser des questions spécifiques sur vos contrats, devis ou sinistres !"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erreur formatage réponse: {e}")
            return "Données récupérées avec succès, mais erreur lors de l'affichage."

async def main():
    """Fonction principale pour test"""
    orchestrator = HybridRAGLangChainOrchestrator()
    
    try:
        await orchestrator.initialize()
        
        # Test avec une requête
        test_query = "Je veux un devis pour une assurance vie"
        result = await orchestrator.process_query(test_query)
        
        print("\n" + "="*50)
        print("RÉSULTAT DU RAG HYBRIDE LANGCHAIN")
        print("="*50)
        print(f"Requête: {result['query']}")
        print(f"Réponse: {result['response']}")
        print(f"Résultats vectoriels: {result['vector_results_count']}")
        print(f"Résultats graphe: {result['graph_results_count']}")
        print(f"Besoin de devis: {result['needs_quotation']}")
        if result['quotation']:
            print(f"Devis: {json.dumps(result['quotation'], indent=2)}")
        
    except Exception as e:
        logger.error(f"❌ Erreur dans main: {e}")

if __name__ == "__main__":
    asyncio.run(main())
