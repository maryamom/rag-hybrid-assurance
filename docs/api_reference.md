# 📚 Référence API

## Endpoints Principaux

### 1. Chat en Temps Réel
**WebSocket** : `ws://localhost:8005/ws`

```javascript
const ws = new WebSocket('ws://localhost:8005/ws');

ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'message',
        content: 'Bonjour, je veux voir mes contrats'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Réponse:', data.response);
};
```

### 2. Génération de Devis
**POST** : `/api/quotation`

```json
{
    "query": "Je veux un devis pour une Peugeot 308, 5 ans d'expérience",
    "user_id": "optional_user_id"
}
```

**Réponse** :
```json
{
    "success": true,
    "quotation": {
        "prime_annuelle": 450.00,
        "prime_mensuelle": 37.50,
        "garanties": [...],
        "options": [...]
    }
}
```

### 3. Authentification
**POST** : `/api/auth/verify`

```json
{
    "email": "client@example.com",
    "code": "123456"
}
```

**Réponse** :
```json
{
    "success": true,
    "token": "jwt_token_here",
    "expires_in": 3600
}
```

### 4. Données Personnelles
**GET** : `/api/personal-data`

**Headers** :
```
Authorization: Bearer jwt_token_here
```

**Réponse** :
```json
{
    "success": true,
    "data": {
        "contrats": [...],
        "devis": [...],
        "sinistres": [...]
    }
}
```

## Codes de Statut

| Code | Description |
|------|-------------|
| 200 | Succès |
| 400 | Requête invalide |
| 401 | Non authentifié |
| 403 | Accès refusé |
| 404 | Ressource non trouvée |
| 500 | Erreur serveur |

## Gestion d'Erreurs

### Format d'Erreur
```json
{
    "error": true,
    "message": "Description de l'erreur",
    "code": "ERROR_CODE",
    "details": {...}
}
```

### Codes d'Erreur
- `AUTH_REQUIRED` : Authentification nécessaire
- `INVALID_CREDENTIALS` : Identifiants invalides
- `CODE_EXPIRED` : Code de vérification expiré
- `QUOTATION_FAILED` : Échec génération devis
- `DATABASE_ERROR` : Erreur base de données

## Rate Limiting

- **Chat** : 60 requêtes/minute
- **Devis** : 10 requêtes/minute
- **Auth** : 5 tentatives/minute

## Webhooks

### Événements Disponibles
- `quotation.generated` : Devis généré
- `auth.success` : Authentification réussie
- `auth.failed` : Échec authentification
- `error.occurred` : Erreur système
