# 🚀 Guide de Déploiement

## Prérequis

### Système
- Docker & Docker Compose
- Python 3.11+ (pour développement)
- 8GB RAM minimum
- 20GB espace disque

### Services Externes
- Compte OpenRouter (DeepSeek v3.1)
- Serveur SMTP (Gmail, SendGrid, etc.)

## Installation Rapide

### 1. Cloner le Repository
```bash
git clone https://github.com/votre-username/rag-hybrid-assurance.git
cd rag-hybrid-assurance
```

### 2. Configuration
```bash
# Copier les fichiers de configuration
cp config/database.yaml.example config/database.yaml
cp config/email.yaml.example config/email.yaml

# Éditer les configurations
nano config/database.yaml
nano config/email.yaml
```

### 3. Variables d'Environnement
```bash
# Créer le fichier .env
cat > .env << EOF
# API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Database Passwords
POSTGRES_PASSWORD=your_secure_password
NEO4J_PASSWORD=your_secure_password
EOF
```

### 4. Déploiement Docker
```bash
# Lancer tous les services
docker-compose up -d

# Vérifier les logs
docker-compose logs -f rag_app

# Vérifier le statut
docker-compose ps
```

## Configuration Détaillée

### Base de Données PostgreSQL
```yaml
# config/database.yaml
postgres:
  host: localhost
  port: 5432
  database: data
  user: postgres
  password: your_secure_password
  pool_size: 10
  max_overflow: 20
```

### Base de Données Neo4j
```yaml
neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: your_secure_password
  max_connection_lifetime: 3600
  max_connection_pool_size: 50
```

### Base de Données Qdrant
```yaml
qdrant:
  host: localhost
  port: 6333
  collection_name: insurance_embeddings
  vector_size: 384
  distance: Cosine
```

## Initialisation des Données

### 1. Préparer les Données
```bash
# Extraire et vectoriser les données
python data_extraction/setup_database.py
python data_extraction/simple_pdf_processor.py
```

### 2. Créer les Collections
```bash
# Créer la collection Qdrant
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
client.create_collection('insurance_embeddings', vector_size=384)
"
```

### 3. Importer les Données
```bash
# Importer les données dans Neo4j
python data_extraction/setup_database.py --neo4j
```

## Monitoring et Maintenance

### Health Checks
```bash
# Vérifier l'état des services
curl http://localhost:8005/health

# Vérifier les logs
docker-compose logs rag_app
docker-compose logs postgres
docker-compose logs neo4j
docker-compose logs qdrant
```

### Sauvegarde
```bash
# Sauvegarder PostgreSQL
docker exec rag_postgres pg_dump -U postgres data > backup_postgres.sql

# Sauvegarder Neo4j
docker exec rag_neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j.dump
docker cp rag_neo4j:/tmp/neo4j.dump ./backup_neo4j.dump

# Sauvegarder Qdrant
docker exec rag_qdrant tar -czf /tmp/qdrant_backup.tar.gz /qdrant/storage
docker cp rag_qdrant:/tmp/qdrant_backup.tar.gz ./backup_qdrant.tar.gz
```

### Mise à Jour
```bash
# Arrêter les services
docker-compose down

# Mettre à jour le code
git pull origin main

# Reconstruire et relancer
docker-compose up -d --build
```

## Production

### Configuration Sécurisée
- Utiliser des mots de passe forts
- Configurer HTTPS/TLS
- Mettre en place un reverse proxy (Nginx)
- Configurer un firewall
- Activer les logs de sécurité

### Scaling
- Utiliser un load balancer
- Mettre en place des réplicas de base de données
- Configurer Redis Cluster
- Utiliser un CDN pour les assets statiques

### Monitoring
- Intégrer Prometheus/Grafana
- Configurer des alertes
- Monitorer les performances
- Surveiller les logs d'erreur
