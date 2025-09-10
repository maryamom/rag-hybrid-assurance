"""
Système d'embeddings vectoriels pour RAG hybride
Création des embeddings vectoriels (FAISS / Chroma)

Objectif: Transformer le texte en vecteurs numériques qui capturent la sémantique
pour permettre la recherche rapide et flexible via similarité vectorielle.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import pool

# Embedding models (Open Source only)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("Warning: sentence-transformers not installed. Please install it for embeddings.")

# Alternative: spaCy embeddings (if available)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False
    print("Info: spaCy not installed. Using only SentenceTransformers for embeddings.")

# Vector databases
try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss not installed. FAISS vector search will not be available.")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None
    print("Warning: chromadb not installed. ChromaDB vector search will not be available.")

# Text processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to simple text splitter
    RecursiveCharacterTextSplitter = None
    print("Warning: langchain not installed. Using simple text splitter.")
import re

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Métadonnées pour chaque chunk de texte"""
    chunk_id: str
    filename: str
    branche: str
    chunk_index: int
    source_type: str = "document"  # document, garantie, contrat, personne, etc.
    start_char: int = 0
    end_char: int = 0
    additional_metadata: Dict[str, Any] = None

@dataclass
class EmbeddingResult:
    """Résultat d'un embedding avec métadonnées"""
    chunk_id: str
    embedding: np.ndarray
    text: str
    metadata: ChunkMetadata

class TextChunker:
    """Classe pour découper le texte en chunks optimisés"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if RecursiveCharacterTextSplitter is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
        else:
            self.text_splitter = None
    
    def chunk_text(self, text: str, filename: str, branche: str, 
                   source_type: str = "document") -> List[Tuple[str, ChunkMetadata]]:
        """
        Découpe le texte en chunks avec métadonnées
        
        Args:
            text: Texte à découper
            filename: Nom du fichier source
            branche: Branche d'assurance
            source_type: Type de source (document, garantie, etc.)
            
        Returns:
            Liste de tuples (chunk_text, metadata)
        """
        if self.text_splitter is not None:
            chunks = self.text_splitter.split_text(text)
        else:
            # Fallback: simple text splitting
            chunks = self._simple_text_split(text)
        
        chunk_data = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{source_type}_{i}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                filename=filename,
                branche=branche,
                chunk_index=i,
                source_type=source_type,
                start_char=i * (self.chunk_size - self.chunk_overlap),
                end_char=min((i + 1) * self.chunk_size, len(text))
            )
            chunk_data.append((chunk, metadata))
        
        return chunk_data
    
    def _simple_text_split(self, text: str) -> List[str]:
        """Fallback text splitting when langchain is not available"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            if last_period > start and last_period > last_newline:
                end = last_period + 1
            elif last_newline > start:
                end = last_newline + 1
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks

class EmbeddingGenerator:
    """Générateur d'embeddings avec support multi-modèles"""
    
    def __init__(self, model_type: str = "sentence_transformers", model_name: str = None):
        self.model_type = model_type
        self.model_name = model_name or self._get_default_model()
        self.model = None
        self.spacy_model = None
        self._initialize_model()
    
    def _get_default_model(self) -> str:
        """Retourne le modèle par défaut selon le type"""
        if self.model_type == "sentence_transformers":
            return "all-MiniLM-L6-v2"  # Fast, good quality
        elif self.model_type == "spacy":
            return "fr_core_news_sm"  # French model
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}. Use 'sentence_transformers' or 'spacy'")
    
    def _initialize_model(self):
        """Initialise le modèle d'embedding (Open Source only)"""
        try:
            if self.model_type == "sentence_transformers":
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers package not installed. Please install it.")
                
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Modèle SentenceTransformers chargé: {self.model_name}")
                
            elif self.model_type == "spacy":
                if not SPACY_AVAILABLE:
                    raise ImportError("spaCy package not installed. Please install it and download a language model.")
                
                try:
                    self.spacy_model = spacy.load(self.model_name)
                    logger.info(f"Modèle spaCy chargé: {self.model_name}")
                except OSError:
                    logger.error(f"Modèle spaCy '{self.model_name}' non trouvé. Installez-le avec: python -m spacy download {self.model_name}")
                    raise
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte (Open Source)"""
        try:
            if self.model_type == "sentence_transformers":
                return self.model.encode(text, convert_to_numpy=True)
                
            elif self.model_type == "spacy":
                doc = self.spacy_model(text)
                return doc.vector
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Génère des embeddings en batch pour optimiser les performances (Open Source)"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.model_type == "sentence_transformers":
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
                
            elif self.model_type == "spacy":
                batch_embeddings = []
                for text in batch:
                    doc = self.spacy_model(text)
                    batch_embeddings.append(doc.vector)
                embeddings.extend(batch_embeddings)
        
        return embeddings

class VectorIndexManager:
    """Gestionnaire d'index vectoriels multi-niveaux"""
    
    def __init__(self, storage_path: str = "./vector_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Indexes FAISS
        self.faiss_indexes = {}
        
        # Base ChromaDB
        if chromadb is not None and Settings is not None:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.storage_path / "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            self.chroma_collections = {}
        else:
            self.chroma_client = None
            self.chroma_collections = {}
            logger.warning("ChromaDB not available. ChromaDB features will be disabled.")
        
        # Métadonnées des embeddings
        self.embeddings_metadata = {}
    
    def create_faiss_index(self, dimension: int, index_name: str, 
                          index_type: str = "flat"):
        """Crée un index FAISS"""
        if faiss is None:
            raise ImportError("FAISS not installed. Please install it or use ChromaDB.")
        
        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)  # Inner Product pour similarité cosinus
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Type d'index FAISS non supporté: {index_type}")
        
        self.faiss_indexes[index_name] = index
        return index
    
    def create_chroma_collection(self, collection_name: str, 
                                metadata: Dict[str, Any] = None):
        """Crée une collection ChromaDB"""
        if self.chroma_client is None:
            raise ImportError("ChromaDB not installed. Please install it or use FAISS.")
        
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            self.chroma_collections[collection_name] = collection
            return collection
        except Exception as e:
            # Collection existe déjà
            collection = self.chroma_client.get_collection(collection_name)
            self.chroma_collections[collection_name] = collection
            return collection
    
    def add_embeddings_to_faiss(self, index_name: str, embeddings: List[np.ndarray], 
                               metadata: List[Dict[str, Any]]):
        """Ajoute des embeddings à un index FAISS"""
        if faiss is None:
            logger.warning("FAISS not available. Skipping FAISS index creation.")
            return
        
        if index_name not in self.faiss_indexes:
            dimension = len(embeddings[0])
            self.create_faiss_index(dimension, index_name)
        
        index = self.faiss_indexes[index_name]
        
        # Normaliser les embeddings pour similarité cosinus
        embeddings_normalized = [emb / np.linalg.norm(emb) for emb in embeddings]
        embeddings_array = np.vstack(embeddings_normalized).astype('float32')
        
        index.add(embeddings_array)
        
        # Sauvegarder les métadonnées
        if index_name not in self.embeddings_metadata:
            self.embeddings_metadata[index_name] = []
        self.embeddings_metadata[index_name].extend(metadata)
        
        # Sauvegarder l'index
        self._save_faiss_index(index_name)
    
    def add_embeddings_to_chroma(self, collection_name: str, 
                                embeddings: List[np.ndarray], 
                                texts: List[str], 
                                metadatas: List[Dict[str, Any]],
                                ids: List[str]):
        """Ajoute des embeddings à une collection ChromaDB"""
        if self.chroma_client is None:
            logger.warning("ChromaDB not available. Skipping ChromaDB collection creation.")
            return
        
        collection = self.create_chroma_collection(collection_name)
        
        collection.add(
            embeddings=[emb.tolist() for emb in embeddings],
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_faiss(self, index_name: str, query_embedding: np.ndarray, 
                    k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Recherche dans un index FAISS"""
        if faiss is None:
            raise ImportError("FAISS not installed. Please install it or use ChromaDB.")
        
        if index_name not in self.faiss_indexes:
            raise ValueError(f"Index {index_name} non trouvé")
        
        index = self.faiss_indexes[index_name]
        
        # Normaliser la requête
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_array = query_normalized.reshape(1, -1).astype('float32')
        
        scores, indices = index.search(query_array, k)
        
        # Récupérer les métadonnées
        results_metadata = []
        for idx in indices[0]:
            if idx < len(self.embeddings_metadata[index_name]):
                results_metadata.append(self.embeddings_metadata[index_name][idx])
        
        return scores[0].tolist(), results_metadata
    
    def search_chroma(self, collection_name: str, query_embedding: np.ndarray, 
                     k: int = 5, where: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recherche dans une collection ChromaDB"""
        if self.chroma_client is None:
            raise ImportError("ChromaDB not installed. Please install it or use FAISS.")
        
        if collection_name not in self.chroma_collections:
            raise ValueError(f"Collection {collection_name} non trouvée")
        
        collection = self.chroma_collections[collection_name]
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where
        )
        
        return results
    
    def _save_faiss_index(self, index_name: str):
        """Sauvegarde un index FAISS"""
        if faiss is None:
            logger.warning("FAISS not available. Cannot save index.")
            return
        
        index_path = self.storage_path / f"{index_name}.faiss"
        metadata_path = self.storage_path / f"{index_name}_metadata.json"
        
        faiss.write_index(self.faiss_indexes[index_name], str(index_path))
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.embeddings_metadata[index_name], f, ensure_ascii=False, indent=2)
    
    def load_faiss_index(self, index_name: str):
        """Charge un index FAISS sauvegardé"""
        if faiss is None:
            logger.warning("FAISS not available. Cannot load index.")
            return
        
        index_path = self.storage_path / f"{index_name}.faiss"
        metadata_path = self.storage_path / f"{index_name}_metadata.json"
        
        if index_path.exists():
            self.faiss_indexes[index_name] = faiss.read_index(str(index_path))
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.embeddings_metadata[index_name] = json.load(f)

class VectorEmbeddingsSystem:
    """Système principal d'embeddings vectoriels"""
    
    def __init__(self, db_config: Dict[str, Any], 
                 embedding_model: str = "sentence_transformers",
                 storage_path: str = "./vector_storage"):
        
        # Configuration base de données
        self.db_config = db_config
        self.connection_pool = None
        
        # Composants du système
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.index_manager = VectorIndexManager(storage_path)
        
        # Configuration des index
        self.index_levels = {
            "document": "index_document",
            "branch": "index_branch", 
            "global": "index_global"
        }
        
        self._initialize_database_pool()
    
    def _initialize_database_pool(self):
        """Initialise le pool de connexions PostgreSQL"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            logger.info("Pool de connexions PostgreSQL initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du pool: {e}")
            raise
    
    def get_database_connection(self):
        """Récupère une connexion du pool"""
        return self.connection_pool.getconn()
    
    def return_database_connection(self, conn):
        """Retourne une connexion au pool"""
        self.connection_pool.putconn(conn)
    
    def load_documents_data(self) -> pd.DataFrame:
        """Charge les données des documents depuis PostgreSQL"""
        conn = self.get_database_connection()
        try:
            query = """
                SELECT filename, branche, text_cleaned
                FROM cleaned_docs
                WHERE text_cleaned IS NOT NULL AND text_cleaned != ''
            """
            df = pd.read_sql(query, conn)
            logger.info(f"Chargé {len(df)} documents depuis cleaned_docs")
            return df
        finally:
            self.return_database_connection(conn)
    
    def load_complementary_tables(self) -> Dict[str, pd.DataFrame]:
        """Charge les tables complémentaires pour enrichissement"""
        conn = self.get_database_connection()
        tables_data = {}
        
        try:
            # Liste des tables complémentaires avec gestion des noms spéciaux
            complementary_tables = [
                "description_des_colonnes_sheet1",
                "description_garanties_sheet1", 
                '"données_assurance_s1.2_s2.2_contrats"',
                '"données_assurance_s1.2_s2.2_garantie_contrat"',
                '"données_assurance_s1.2_s2.2_mapping_produits"',
                '"données_assurance_s1.2_s2.2_personne_morale"',
                '"données_assurance_s1.2_s2.2_personne_physique"',
                '"données_assurance_s1.2_s2.2_sinistres"',
                "exemples_questions_reponses_qr",
                "mapping_produits_vs_profils_cibles_sheet1"
            ]
            
            for table_name in complementary_tables:
                try:
                    # Nettoyer le nom de table pour l'utilisation dans le code
                    clean_table_name = table_name.strip('"')
                    query = f"SELECT * FROM {table_name}"
                    df = pd.read_sql(query, conn)
                    tables_data[clean_table_name] = df
                    logger.info(f"Chargé {len(df)} lignes depuis {clean_table_name}")
                except Exception as e:
                    logger.warning(f"Impossible de charger {table_name}: {e}")
                    
        finally:
            self.return_database_connection(conn)
        
        return tables_data
    
    def create_document_chunks(self, documents_df: pd.DataFrame) -> List[Tuple[str, ChunkMetadata]]:
        """Crée les chunks pour tous les documents"""
        all_chunks = []
        
        for _, row in documents_df.iterrows():
            filename = row['filename']
            branche = row['branche']
            text = row['text_cleaned']
            
            if pd.isna(text) or text.strip() == "":
                continue
            
            chunks = self.chunker.chunk_text(text, filename, branche, "document")
            all_chunks.extend(chunks)
            
            logger.info(f"Créé {len(chunks)} chunks pour {filename}")
        
        logger.info(f"Total: {len(all_chunks)} chunks créés")
        return all_chunks
    
    def create_complementary_chunks(self, tables_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks à partir des tables complémentaires"""
        complementary_chunks = []
        
        for table_name, df in tables_data.items():
            if df.empty:
                continue
            
            # Créer des chunks selon le type de table
            if "garantie" in table_name.lower():
                chunks = self._create_guarantee_chunks(df, table_name)
            elif "contrat" in table_name.lower():
                chunks = self._create_contract_chunks(df, table_name)
            elif "personne" in table_name.lower():
                chunks = self._create_person_chunks(df, table_name)
            elif "sinistre" in table_name.lower():
                chunks = self._create_claim_chunks(df, table_name)
            elif "question" in table_name.lower():
                chunks = self._create_qa_chunks(df, table_name)
            else:
                chunks = self._create_generic_chunks(df, table_name)
            
            complementary_chunks.extend(chunks)
            logger.info(f"Créé {len(chunks)} chunks depuis {table_name}")
        
        return complementary_chunks
    
    def _create_guarantee_chunks(self, df: pd.DataFrame, table_name: str) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks pour les garanties"""
        chunks = []
        
        for _, row in df.iterrows():
            # Construire un texte descriptif de la garantie
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(f"{col}: {row[col]}")
            
            if text_parts:
                text = " | ".join(text_parts)
                chunk_id = f"{table_name}_{row.name}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=table_name,
                    branche="GARANTIES",
                    chunk_index=row.name,
                    source_type="garantie"
                )
                chunks.append((text, metadata))
        
        return chunks
    
    def _create_contract_chunks(self, df: pd.DataFrame, table_name: str) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks pour les contrats"""
        chunks = []
        
        for _, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(f"{col}: {row[col]}")
            
            if text_parts:
                text = " | ".join(text_parts)
                chunk_id = f"{table_name}_{row.name}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=table_name,
                    branche="CONTRATS",
                    chunk_index=row.name,
                    source_type="contrat"
                )
                chunks.append((text, metadata))
        
        return chunks
    
    def _create_person_chunks(self, df: pd.DataFrame, table_name: str) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks pour les personnes"""
        chunks = []
        
        for _, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(f"{col}: {row[col]}")
            
            if text_parts:
                text = " | ".join(text_parts)
                chunk_id = f"{table_name}_{row.name}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=table_name,
                    branche="PERSONNES",
                    chunk_index=row.name,
                    source_type="personne"
                )
                chunks.append((text, metadata))
        
        return chunks
    
    def _create_claim_chunks(self, df: pd.DataFrame, table_name: str) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks pour les sinistres"""
        chunks = []
        
        for _, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(f"{col}: {row[col]}")
            
            if text_parts:
                text = " | ".join(text_parts)
                chunk_id = f"{table_name}_{row.name}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=table_name,
                    branche="SINISTRES",
                    chunk_index=row.name,
                    source_type="sinistre"
                )
                chunks.append((text, metadata))
        
        return chunks
    
    def _create_qa_chunks(self, df: pd.DataFrame, table_name: str) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks pour les questions-réponses"""
        chunks = []
        
        for _, row in df.iterrows():
            if 'question' in df.columns and 'reponse' in df.columns:
                text = f"Question: {row['question']} | Réponse: {row['reponse']}"
            else:
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        text_parts.append(f"{col}: {row[col]}")
                text = " | ".join(text_parts)
            
            if text.strip():
                chunk_id = f"{table_name}_{row.name}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=table_name,
                    branche="FAQ",
                    chunk_index=row.name,
                    source_type="qa"
                )
                chunks.append((text, metadata))
        
        return chunks
    
    def _create_generic_chunks(self, df: pd.DataFrame, table_name: str) -> List[Tuple[str, ChunkMetadata]]:
        """Crée des chunks génériques pour les autres tables"""
        chunks = []
        
        for _, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(f"{col}: {row[col]}")
            
            if text_parts:
                text = " | ".join(text_parts)
                chunk_id = f"{table_name}_{row.name}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=table_name,
                    branche="AUTRES",
                    chunk_index=row.name,
                    source_type="table"
                )
                chunks.append((text, metadata))
        
        return chunks
    
    def generate_embeddings_for_chunks(self, chunks: List[Tuple[str, ChunkMetadata]], 
                                     batch_size: int = 32) -> List[EmbeddingResult]:
        """Génère les embeddings pour une liste de chunks"""
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        
        logger.info(f"Génération des embeddings pour {len(texts)} chunks...")
        
        embeddings = self.embedding_generator.generate_embeddings_batch(texts, batch_size)
        
        results = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            result = EmbeddingResult(
                chunk_id=metadata.chunk_id,
                embedding=embedding,
                text=text,
                metadata=metadata
            )
            results.append(result)
        
        logger.info(f"Généré {len(results)} embeddings")
        return results
    
    def create_multi_level_indexes(self, embedding_results: List[EmbeddingResult]):
        """Crée les index vectoriels multi-niveaux"""
        logger.info("Création des index vectoriels multi-niveaux...")
        
        # Index global
        self._create_global_index(embedding_results)
        
        # Index par branche
        self._create_branch_indexes(embedding_results)
        
        # Index par document
        self._create_document_indexes(embedding_results)
        
        logger.info("Index vectoriels créés avec succès")
    
    def _create_global_index(self, embedding_results: List[EmbeddingResult]):
        """Crée l'index global"""
        embeddings = [result.embedding for result in embedding_results]
        metadatas = [self._metadata_to_dict(result.metadata) for result in embedding_results]
        
        self.index_manager.add_embeddings_to_faiss(
            self.index_levels["global"], embeddings, metadatas
        )
        
        # Créer aussi une collection ChromaDB globale
        texts = [result.text for result in embedding_results]
        ids = [result.chunk_id for result in embedding_results]
        
        self.index_manager.add_embeddings_to_chroma(
            "global_collection", embeddings, texts, metadatas, ids
        )
        
        logger.info(f"Index global créé avec {len(embedding_results)} embeddings")
    
    def _create_branch_indexes(self, embedding_results: List[EmbeddingResult]):
        """Crée les index par branche"""
        branches = {}
        
        for result in embedding_results:
            branche = result.metadata.branche
            if branche not in branches:
                branches[branche] = []
            branches[branche].append(result)
        
        for branche, results in branches.items():
            embeddings = [result.embedding for result in results]
            metadatas = [self._metadata_to_dict(result.metadata) for result in results]
            texts = [result.text for result in results]
            ids = [result.chunk_id for result in results]
            
            # Index FAISS par branche
            index_name = f"branch_{branche.replace(' ', '_')}"
            self.index_manager.add_embeddings_to_faiss(index_name, embeddings, metadatas)
            
            # Collection ChromaDB par branche
            collection_name = f"branch_{branche.replace(' ', '_')}"
            self.index_manager.add_embeddings_to_chroma(
                collection_name, embeddings, texts, metadatas, ids
            )
            
            logger.info(f"Index branche '{branche}' créé avec {len(results)} embeddings")
    
    def _create_document_indexes(self, embedding_results: List[EmbeddingResult]):
        """Crée les index par document"""
        documents = {}
        
        for result in embedding_results:
            filename = result.metadata.filename
            if filename not in documents:
                documents[filename] = []
            documents[filename].append(result)
        
        for filename, results in documents.items():
            embeddings = [result.embedding for result in results]
            metadatas = [self._metadata_to_dict(result.metadata) for result in results]
            texts = [result.text for result in results]
            ids = [result.chunk_id for result in results]
            
            # Index FAISS par document
            index_name = f"doc_{filename.replace('.', '_').replace(' ', '_')}"
            self.index_manager.add_embeddings_to_faiss(index_name, embeddings, metadatas)
            
            # Collection ChromaDB par document
            collection_name = f"doc_{filename.replace('.', '_').replace(' ', '_')}"
            self.index_manager.add_embeddings_to_chroma(
                collection_name, embeddings, texts, metadatas, ids
            )
            
            logger.info(f"Index document '{filename}' créé avec {len(results)} embeddings")
    
    def _metadata_to_dict(self, metadata: ChunkMetadata) -> Dict[str, Any]:
        """Convertit les métadonnées en dictionnaire"""
        return {
            "chunk_id": metadata.chunk_id,
            "filename": metadata.filename,
            "branche": metadata.branche,
            "chunk_index": metadata.chunk_index,
            "source_type": metadata.source_type,
            "start_char": metadata.start_char,
            "end_char": metadata.end_char,
            "additional_metadata": metadata.additional_metadata or {}
        }
    
    def build_complete_system(self):
        """Construit le système complet d'embeddings vectoriels"""
        logger.info("Début de la construction du système d'embeddings vectoriels...")
        
        # 1. Charger les données
        logger.info("1. Chargement des données...")
        documents_df = self.load_documents_data()
        complementary_tables = self.load_complementary_tables()
        
        # 2. Créer les chunks
        logger.info("2. Création des chunks...")
        document_chunks = self.create_document_chunks(documents_df)
        complementary_chunks = self.create_complementary_chunks(complementary_tables)
        
        all_chunks = document_chunks + complementary_chunks
        logger.info(f"Total chunks créés: {len(all_chunks)}")
        
        # 3. Générer les embeddings
        logger.info("3. Génération des embeddings...")
        embedding_results = self.generate_embeddings_for_chunks(all_chunks)
        
        # 4. Créer les index
        logger.info("4. Création des index vectoriels...")
        self.create_multi_level_indexes(embedding_results)
        
        logger.info("Système d'embeddings vectoriels construit avec succès!")
        return embedding_results
    
    def semantic_search(self, query: str, level: str = "global", 
                       k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Effectue une recherche sémantique
        
        Args:
            query: Requête textuelle
            level: Niveau de recherche (global, branch, document)
            k: Nombre de résultats
            filters: Filtres de métadonnées
            
        Returns:
            Liste des résultats avec scores et métadonnées
        """
        # Générer l'embedding de la requête
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        results = []
        
        if level == "global":
            # Recherche dans l'index global
            scores, metadatas = self.index_manager.search_faiss(
                self.index_levels["global"], query_embedding, k
            )
            
            for score, metadata in zip(scores, metadatas):
                results.append({
                    "score": float(score),
                    "metadata": metadata,
                    "level": "global"
                })
        
        elif level == "branch" and filters and "branche" in filters:
            # Recherche dans l'index de branche spécifique
            branche = filters["branche"]
            index_name = f"branch_{branche.replace(' ', '_')}"
            
            try:
                scores, metadatas = self.index_manager.search_faiss(
                    index_name, query_embedding, k
                )
                
                for score, metadata in zip(scores, metadatas):
                    results.append({
                        "score": float(score),
                        "metadata": metadata,
                        "level": "branch"
                    })
            except ValueError:
                logger.warning(f"Index de branche '{branche}' non trouvé")
        
        elif level == "document" and filters and "filename" in filters:
            # Recherche dans l'index de document spécifique
            filename = filters["filename"]
            index_name = f"doc_{filename.replace('.', '_').replace(' ', '_')}"
            
            try:
                scores, metadatas = self.index_manager.search_faiss(
                    index_name, query_embedding, k
                )
                
                for score, metadata in zip(scores, metadatas):
                    results.append({
                        "score": float(score),
                        "metadata": metadata,
                        "level": "document"
                    })
            except ValueError:
                logger.warning(f"Index de document '{filename}' non trouvé")
        
        return results

def main():
    """Fonction principale pour construire le système d'embeddings"""
    
    # Configuration de la base de données
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'data',
        'user': 'postgres',
        'password': 'Maryembo3'
    }
    
    # Créer le système d'embeddings (Open Source)
    vector_system = VectorEmbeddingsSystem(
        db_config=db_config,
        embedding_model="sentence_transformers",  # Open source, no API key needed
        storage_path="./vector_storage"
    )
    
    # Construire le système complet
    embedding_results = vector_system.build_complete_system()
    
    # Exemple de recherche sémantique
    print("\n=== Exemple de recherche sémantique ===")
    query = "garantie décès accident"
    results = vector_system.semantic_search(query, level="global", k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nRésultat {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Fichier: {result['metadata']['filename']}")
        print(f"Branche: {result['metadata']['branche']}")
        print(f"Type: {result['metadata']['source_type']}")

if __name__ == "__main__":
    main()
