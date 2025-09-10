# 🤝 Guide de Contribution

Merci de votre intérêt à contribuer au projet RAG Hybrid Assurance ! Ce guide vous aidera à contribuer efficacement.

## 🚀 Démarrage Rapide

### 1. Fork et Clone
```bash
# Fork le repository sur GitHub
# Puis cloner votre fork
git clone https://github.com/votre-username/rag-hybrid-assurance.git
cd rag-hybrid-assurance

# Ajouter le repository original comme remote
git remote add upstream https://github.com/original-username/rag-hybrid-assurance.git
```

### 2. Configuration de l'Environnement
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer les dépendances de développement
pip install -r requirements-dev.txt
```

## 📋 Types de Contributions

### 🐛 Bug Reports
- Utilisez le template d'issue pour les bugs
- Incluez des étapes de reproduction
- Ajoutez des logs d'erreur si possible

### ✨ Nouvelles Fonctionnalités
- Ouvrez d'abord une issue pour discuter
- Créez une branche feature
- Suivez les conventions de code

### 📚 Documentation
- Améliorez la documentation existante
- Ajoutez des exemples d'utilisation
- Corrigez les erreurs de typo

### 🧪 Tests
- Ajoutez des tests pour les nouvelles fonctionnalités
- Améliorez la couverture de tests
- Corrigez les tests cassés

## 🔧 Workflow de Développement

### 1. Créer une Branche
```bash
git checkout -b feature/nom-de-la-fonctionnalite
# ou
git checkout -b bugfix/description-du-bug
```

### 2. Développer
- Écrivez du code propre et commenté
- Suivez les conventions de style
- Ajoutez des tests si nécessaire

### 3. Tester
```bash
# Lancer les tests
pytest tests/

# Vérifier le style de code
black src/
flake8 src/

# Lancer les tests de linting
pylint src/
```

### 4. Commit
```bash
git add .
git commit -m "feat: ajouter nouvelle fonctionnalité"
# ou
git commit -m "fix: corriger bug dans l'authentification"
```

### 5. Push et Pull Request
```bash
git push origin feature/nom-de-la-fonctionnalite
# Puis créer une Pull Request sur GitHub
```

## 📝 Conventions de Code

### Style Python
- Utilisez `black` pour le formatage
- Suivez PEP 8
- Utilisez des docstrings pour les fonctions
- Nommez les variables de manière descriptive

### Messages de Commit
Utilisez le format conventionnel :
- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `docs:` documentation
- `style:` formatage, pas de changement de logique
- `refactor:` refactoring de code
- `test:` ajout de tests
- `chore:` maintenance

### Structure des Fichiers
```
src/
├── core/           # Modules centraux
├── auth/           # Authentification
├── api/            # Interface API
└── utils/          # Utilitaires

tests/
├── unit/           # Tests unitaires
├── integration/    # Tests d'intégration
└── e2e/           # Tests end-to-end
```

## 🧪 Tests

### Types de Tests
- **Unitaires** : Test des fonctions individuelles
- **Intégration** : Test des interactions entre composants
- **End-to-End** : Test du flux complet

### Lancer les Tests
```bash
# Tous les tests
pytest

# Tests spécifiques
pytest tests/unit/test_auth.py

# Avec couverture
pytest --cov=src tests/
```

## 📖 Documentation

### Ajouter de la Documentation
- Mettez à jour le README si nécessaire
- Ajoutez des docstrings aux nouvelles fonctions
- Mettez à jour la documentation API
- Ajoutez des exemples d'utilisation

### Format de Documentation
```python
def nouvelle_fonction(param1: str, param2: int) -> dict:
    """
    Description courte de la fonction.
    
    Args:
        param1: Description du premier paramètre
        param2: Description du deuxième paramètre
        
    Returns:
        Description de la valeur de retour
        
    Raises:
        ValueError: Quand param1 est vide
        
    Example:
        >>> result = nouvelle_fonction("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

## 🔍 Review Process

### Avant de Soumettre
- [ ] Code testé localement
- [ ] Tests passent
- [ ] Documentation mise à jour
- [ ] Pas de conflits de merge
- [ ] Messages de commit clairs

### Pendant la Review
- Répondez aux commentaires
- Faites les modifications demandées
- Soyez ouvert aux suggestions
- Posez des questions si nécessaire

## 🆘 Aide

### Questions
- Ouvrez une issue pour les questions
- Utilisez les discussions GitHub
- Contactez les mainteneurs

### Ressources
- [Documentation LangChain](https://python.langchain.com/)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Guide Qdrant](https://qdrant.tech/documentation/)

## 🙏 Reconnaissance

Tous les contributeurs seront mentionnés dans le README du projet.

Merci de contribuer ! 🎉
