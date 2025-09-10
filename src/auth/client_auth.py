#!/usr/bin/env python3
"""
Système d'authentification client pour accès aux données personnelles
"""

import asyncio
import smtplib
import random
import string
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientAuthSystem:
    """Système d'authentification client"""
    
    def __init__(self, db_config: Dict[str, str], smtp_config: Dict[str, str]):
        self.db_config = db_config
        self.smtp_config = smtp_config
        self.active_codes = {}  # {email: {code, timestamp, client_id}}
        self.session_tokens = {}  # {token: {client_id, timestamp}}
        
    def generate_verification_code(self) -> str:
        """Génère un code de vérification à 6 chiffres"""
        return ''.join(random.choices(string.digits, k=6))
    
    def generate_session_token(self) -> str:
        """Génère un token de session"""
        return hashlib.sha256(f"{time.time()}{random.random()}".encode()).hexdigest()[:32]
    
    async def find_client_by_identity(self, nom: str, prenom: str, telephone: str) -> Optional[Dict[str, Any]]:
        """Recherche un client par nom, prénom et téléphone"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Recherche dans la table clients
            query = """
            SELECT * FROM "Data_donnees_assurance_s1_clients" 
            WHERE LOWER(nom) = LOWER(%s) 
            AND LOWER(prenom) = LOWER(%s) 
            AND telephone = %s
            """
            
            cursor.execute(query, (nom, prenom, telephone))
            client = cursor.fetchone()
            
            if client:
                logger.info(f"✅ Client trouvé: {nom} {prenom}")
                return dict(client)
            else:
                logger.warning(f"❌ Client non trouvé: {nom} {prenom} - {telephone}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur recherche client: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    async def send_verification_email(self, email: str, code: str, nom: str, prenom: str) -> bool:
        """Envoie un email de vérification"""
        try:
            subject = "Code de vérification - BHbot Assurance"
            body = f"""
            Bonjour {prenom} {nom},
            
            Voici votre code de vérification pour accéder à vos informations d'assurance :
            
            🔐 Code de vérification : {code}
            
            Ce code est valide pendant 10 minutes.
            
            Si vous n'avez pas demandé cette vérification, ignorez cet email.
            
            Cordialement,
            L'équipe BHbot Assurance
            """
            
            # Configuration SMTP
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['user'], self.smtp_config['password'])
            
            # Envoi de l'email
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(
                self.smtp_config['user'],
                email,
                message.encode('utf-8')
            )
            server.quit()
            
            logger.info(f"✅ Email envoyé à {email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi email: {e}")
            return False
    
    async def request_client_verification(self, nom: str, prenom: str, telephone: str, email: str) -> Dict[str, Any]:
        """Demande de vérification client"""
        try:
            # 1. Rechercher le client
            client = await self.find_client_by_identity(nom, prenom, telephone)
            
            if not client:
                return {
                    'success': False,
                    'message': 'Client non trouvé. Vérifiez vos informations.',
                    'error_type': 'client_not_found'
                }
            
            # 2. Générer le code de vérification
            code = self.generate_verification_code()
            
            # 3. Envoyer l'email
            email_sent = await self.send_verification_email(email, code, nom, prenom)
            
            if not email_sent:
                return {
                    'success': False,
                    'message': 'Erreur lors de l\'envoi de l\'email.',
                    'error_type': 'email_error'
                }
            
            # 4. Stocker le code temporairement
            self.active_codes[email] = {
                'code': code,
                'timestamp': time.time(),
                'client_id': client['id'],
                'client_data': client
            }
            
            return {
                'success': True,
                'message': f'Code de vérification envoyé à {email}',
                'email': email,
                'expires_in': 600  # 10 minutes
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur demande vérification: {e}")
            return {
                'success': False,
                'message': 'Erreur interne du système.',
                'error_type': 'internal_error'
            }
    
    async def verify_code(self, email: str, code: str) -> Dict[str, Any]:
        """Vérifie le code de vérification"""
        try:
            if email not in self.active_codes:
                return {
                    'success': False,
                    'message': 'Aucun code en attente pour cet email.',
                    'error_type': 'no_pending_code'
                }
            
            stored_data = self.active_codes[email]
            
            # Vérifier l'expiration (10 minutes)
            if time.time() - stored_data['timestamp'] > 600:
                del self.active_codes[email]
                return {
                    'success': False,
                    'message': 'Code expiré. Veuillez redemander une vérification.',
                    'error_type': 'code_expired'
                }
            
            # Vérifier le code
            if stored_data['code'] != code:
                return {
                    'success': False,
                    'message': 'Code incorrect.',
                    'error_type': 'invalid_code'
                }
            
            # Code valide - créer une session
            session_token = self.generate_session_token()
            self.session_tokens[session_token] = {
                'client_id': stored_data['client_id'],
                'timestamp': time.time(),
                'client_data': stored_data['client_data']
            }
            
            # Nettoyer le code utilisé
            del self.active_codes[email]
            
            return {
                'success': True,
                'message': 'Authentification réussie !',
                'session_token': session_token,
                'client_name': f"{stored_data['client_data']['prenom']} {stored_data['client_data']['nom']}"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification code: {e}")
            return {
                'success': False,
                'message': 'Erreur interne du système.',
                'error_type': 'internal_error'
            }
    
    async def get_client_data(self, session_token: str) -> Dict[str, Any]:
        """Récupère les données du client authentifié"""
        try:
            if session_token not in self.session_tokens:
                return {
                    'success': False,
                    'message': 'Session invalide. Veuillez vous réauthentifier.',
                    'error_type': 'invalid_session'
                }
            
            session_data = self.session_tokens[session_token]
            client_id = session_data['client_id']
            
            # Vérifier l'expiration de la session (1 heure)
            if time.time() - session_data['timestamp'] > 3600:
                del self.session_tokens[session_token]
                return {
                    'success': False,
                    'message': 'Session expirée. Veuillez vous réauthentifier.',
                    'error_type': 'session_expired'
                }
            
            # Récupérer les données du client
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Informations personnelles
            client_query = """
            SELECT * FROM "Data_donnees_assurance_s1_clients" 
            WHERE id = %s
            """
            cursor.execute(client_query, (client_id,))
            client_info = dict(cursor.fetchone())
            
            # Devis du client
            devis_query = """
            SELECT * FROM "Data_donnees_assurance_s1_devis" 
            WHERE client_id = %s
            ORDER BY created_at DESC
            """
            cursor.execute(devis_query, (client_id,))
            devis = [dict(row) for row in cursor.fetchall()]
            
            # Contrats du client
            contrats_query = """
            SELECT * FROM "Data_donnees_assurance_s1_contrats" 
            WHERE client_id = %s
            ORDER BY dateeffet DESC
            """
            cursor.execute(contrats_query, (client_id,))
            contrats = [dict(row) for row in cursor.fetchall()]
            
            return {
                'success': True,
                'client_info': client_info,
                'devis': devis,
                'contrats': contrats,
                'summary': {
                    'total_devis': len(devis),
                    'total_contrats': len(contrats),
                    'last_devis': devis[0]['created_at'] if devis else None,
                    'last_contrat': contrats[0]['dateeffet'] if contrats else None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données: {e}")
            return {
                'success': False,
                'message': 'Erreur lors de la récupération des données.',
                'error_type': 'data_error'
            }
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    async def logout(self, session_token: str) -> Dict[str, Any]:
        """Déconnexion du client"""
        if session_token in self.session_tokens:
            del self.session_tokens[session_token]
            return {
                'success': True,
                'message': 'Déconnexion réussie.'
            }
        else:
            return {
                'success': False,
                'message': 'Session invalide.'
            }

# Configuration par défaut
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'assurance_db',
    'user': 'postgres',
    'password': 'password'
}

DEFAULT_SMTP_CONFIG = {
    'host': 'smtp.gmail.com',
    'port': 587,
    'user': 'your-email@gmail.com',
    'password': 'your-app-password'
}

# Instance globale
auth_system = ClientAuthSystem(DEFAULT_DB_CONFIG, DEFAULT_SMTP_CONFIG)
