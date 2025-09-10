# 🏗️ Architecture Technique Détaillée

## Vue d'Ensemble

Le système RAG hybride pour l'assurance combine plusieurs technologies d'IA pour fournir des réponses intelligentes et contextuelles aux clients.

## Composants Principaux

### 1. LangChain Orchestrator
- **Rôle** : Chef d'orchestre central du système
- **Fonctions** :
  - Coordination des retrievers
  - Gestion de la mémoire conversationnelle
  - Ingénierie des prompts
  - Chaînes de traitement

### 2. Recherche Vectorielle (Qdrant)
- **Base de données** : Qdrant
- **Contenu** : Documents PDF, contrats, garanties
- **Technologie** : Embeddings Sentence Transformers
- **Recherche** : Similarité cosinus

### 3. Recherche Relationnelle (Neo4j)
- **Base de données** : Neo4j Graph DB
- **Contenu** : Relations clients, contrats, sinistres
- **Technologie** : Requêtes Cypher
- **Recherche** : Traversal de graphe

### 4. Génération Intelligente (DeepSeek v3.1)
- **Modèle** : DeepSeek v3.1 via OpenRouter
- **Fonction** : Génération de réponses naturelles
- **Contexte** : Fusion des résultats vectoriel + graphe

## Pipeline de Traitement

```
User Query
    ↓
FastAPI Interface (Port 8005)
    ↓
LangChain Orchestrator
    ├── Intent Classification
    ├── Entity Extraction
    └── Service Routing
    ↓
Parallel Retrieval
    ├── Vector Search (Qdrant)
    └── Graph Search (Neo4j)
    ↓
Context Assembly
    ├── Merge Results
    ├── Deduplication
    └── Ranking
    ↓
Response Generation
    ├── DeepSeek v3.1
    ├── Specialized Handlers
    └── Formatting
    ↓
Contextually Rich Response
```

## Sécurité

### Authentification Multi-Étapes
1. **Validation des données** : Nom, prénom, téléphone
2. **Vérification en base** : Recherche PostgreSQL
3. **Code de vérification** : Envoi par email sécurisé
4. **Validation du code** : Vérification avec expiration
5. **Accès aux données** : Récupération depuis Neo4j

### Technologies de Sécurité
- **JWT Tokens** : Authentification stateless
- **bcrypt** : Hachage des mots de passe
- **TLS/SSL** : Chiffrement des communications
- **Rate Limiting** : Protection contre les attaques
- **Validation stricte** : Protection contre les injections

## Performance

### Optimisations
- **Cache Redis** : Mise en cache des résultats fréquents
- **Connection Pooling** : Pool de connexions base de données
- **Requêtes parallèles** : Recherche simultanée vectoriel + graphe
- **Indexation optimisée** : Index sur les champs critiques

### Métriques
- **Recherche vectorielle** : < 100ms
- **Recherche graphe** : < 200ms
- **Génération de réponse** : < 2s
- **Temps total** : < 3s

## Déploiement

### Docker Compose
```bash
docker-compose up -d
```

### Services Inclus
- PostgreSQL (Port 5432)
- Neo4j (Ports 7474, 7687)
- Qdrant (Ports 6333, 6334)
- Redis (Port 6379)
- RAG Application (Port 8005)

### Configuration
- Variables d'environnement
- Fichiers de configuration YAML
- Secrets management
- Health checks
