#  RAG Hybride pour l'Assurance

##  Description

Solution RAG (Retrieval-Augmented Generation) hybride intelligente pour les compagnies d'assurance, combinant recherche sémantique et relationnelle pour fournir des réponses précises et contextuelles aux clients.

##  Architecture

### Technologies Principales
- **LangChain** : Orchestrateur central du système
- **Qdrant** : Base de données vectorielle pour la recherche sémantique
- **Neo4j** : Base de données graphe pour les relations
- **DeepSeek v3.1** : Modèle de langage pour la génération
- **PostgreSQL** : Base de données métier
- **FastAPI** : Interface API REST

### Pipeline RAG Hybride
```
User Query → LangChain Orchestrator → [Vector Search (Qdrant) + Graph Search (Neo4j)] → Context Assembly → DeepSeek v3.1 → Response
```

##  Fonctionnalités

### ✅ Recherche Intelligente
- **Recherche Sémantique** : Comprend le sens des questions via embeddings
- **Recherche Relationnelle** : Trouve les connexions entre données via graphes
- **Fusion Hybride** : Combine intelligemment les résultats

### ✅ Accès Sécurisé aux Données Personnelles
- Authentification multi-étapes (nom, prénom, téléphone)
- Vérification par email avec code sécurisé
- Accès aux contrats, devis, sinistres personnels

### ✅ Génération de Devis Automatique
- Extraction automatique des paramètres depuis le texte libre
- Intégration avec API externe de devis
- Système de fallback pour la continuité du service

### ✅ Interface Utilisateur Moderne
- Chat en temps réel via WebSocket
- Interface web responsive
- Génération de devis interactifs

## 📁 Structure du Projet

```
├── src/                          # Code source principal
│   ├── core/                     # Modules centraux
│   │   ├── orchestrator.py       # Orchestrateur LangChain
│   │   ├── retrievers.py         # Retrievers vectoriel et graphe
│   │   └── generators.py         # Générateurs de réponses
│   ├── auth/                     # Système d'authentification
│   │   ├── personal_data_auth.py # Auth données personnelles
│   │   └── client_auth.py        # Auth clients
│   ├── api/                      # Interface API
│   │   ├── main.py              # FastAPI principal
│   │   └── routes/              # Routes API
│   └── utils/                    # Utilitaires
├── data_extraction/              # Extraction et vectorisation
│   ├── pdf_processor.py         # Traitement PDF
│   ├── vectorizer.py            # Vectorisation données
│   └── database_setup.py        # Configuration BDD
├── docs/                         # Documentation
│   ├── architecture.md          # Architecture détaillée
│   ├── api_reference.md         # Référence API
│   └── deployment.md            # Guide déploiement
├── config/                       # Configuration
│   ├── database.yaml            # Config bases de données
│   ├── email.yaml               # Config email
│   └── models.yaml              # Config modèles IA
├── tests/                        # Tests
├── requirements.txt              # Dépendances Python
└── docker-compose.yml           # Déploiement Docker
```

## 🛠️ Installation

### Prérequis
- Python 3.11+
- PostgreSQL
- Neo4j
- Qdrant
- Docker (optionnel)

### Installation Rapide
```bash
# Cloner le repository
git clone https://github.com/votre-username/rag-hybrid-assurance.git
cd rag-hybrid-assurance

# Installer les dépendances
pip install -r requirements.txt

# Configurer les bases de données
cp config/database.yaml.example config/database.yaml
# Éditer les configurations

# Lancer l'application
python src/api/main.py
```

## 🔧 Configuration

### Variables d'Environnement
```bash
# Bases de données
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

QDRANT_HOST=localhost
QDRANT_PORT=6333

# API Keys
DEEPSEEK_API_KEY=your_api_key
OPENROUTER_API_KEY=your_api_key

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

## 📊 Utilisation

### 1. Questions Générales
```
Utilisateur: "Qu'est-ce que l'assurance auto ?"
Système: [Réponse basée sur les documents vectorisés]
```

### 2. Données Personnelles
```
Utilisateur: "Montrez-moi mes contrats"
Système: [Demande d'authentification → Code email → Affichage des données]
```

### 3. Génération de Devis
```
Utilisateur: "Je veux un devis pour une Peugeot 308, 5 ans d'expérience"
Système: [Génération automatique du devis via API externe]
```

## 🔒 Sécurité

- **Authentification JWT** avec expiration
- **Chiffrement TLS/SSL** pour les communications
- **Validation stricte** des entrées utilisateur
- **Rate limiting** pour prévenir les attaques
- **Codes de vérification** à usage unique

## 📈 Performance

- **Recherche vectorielle** : < 100ms
- **Recherche graphe** : < 200ms
- **Génération de réponse** : < 2s
- **Cache Redis** pour optimiser les performances

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

Pour toute question ou support :
- 📧 Email : support@rag-assurance.com
- 🐛 Issues : [GitHub Issues](https://github.com/votre-username/rag-hybrid-assurance/issues)
- 📖 Documentation : [Wiki du projet](https://github.com/votre-username/rag-hybrid-assurance/wiki)

## 🙏 Remerciements

- [LangChain](https://langchain.com/) pour l'orchestration
- [Qdrant](https://qdrant.tech/) pour la recherche vectorielle
- [Neo4j](https://neo4j.com/) pour la base de données graphe
- [DeepSeek](https://deepseek.com/) pour le modèle de langage
