#!/usr/bin/env python3
"""
Système d'authentification pour les données personnelles
Vérifie l'email et envoie un code de vérification
"""

import smtplib
import random
import string
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import json
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalDataAuthSystem:
    """Système d'authentification pour les données personnelles"""
    
    def __init__(self):
        self.qdrant_config = {
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'insurance_embeddings'
        }
        
        # Configuration PostgreSQL
        self.postgres_config = {
            'host': 'localhost',
            'port': '5432',
            'dbname': 'data',
            'user': 'postgres',
            'password': 'Maryembo3'
        }
        
        # Configuration email (à personnaliser)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',  # Ou votre serveur SMTP
            'smtp_port': 587,
            'email': 'votre-email@gmail.com',  # Votre email
            'password': 'votre-mot-de-passe-app',  # Mot de passe d'application
            'from_name': 'Système Assurance RAG'
        }
        
        # Stockage temporaire des codes (en production, utiliser Redis/DB)
        self.verification_codes = {}
        
        # Détection des mots-clés pour données personnelles
        self.personal_data_keywords = [
            'mon contrat', 'mes données', 'mon assurance', 'mes informations',
            'mon profil', 'mes garanties', 'mon sinistre', 'mon devis',
            'mes contrats', 'mes polices', 'mon historique', 'mes paiements'
        ]
    
    def detect_personal_data_request(self, query: str) -> bool:
        """Détecte si la requête concerne des données personnelles"""
        query_lower = query.lower()
        
        # Vérifier les mots-clés
        for keyword in self.personal_data_keywords:
            if keyword in query_lower:
                return True
        
        # Vérifier les pronoms possessifs
        possessive_pronouns = ['mon', 'ma', 'mes', 'mon', 'ma', 'mes']
        for pronoun in possessive_pronouns:
            if pronoun in query_lower:
                return True
        
        return False
    
    def search_email_in_database(self, email: str) -> Tuple[bool, List[Dict]]:
        """Recherche l'email dans la base de données PostgreSQL"""
        try:
            import psycopg2
            
            connection = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                dbname=self.postgres_config['dbname'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                client_encoding='utf8'
            )
            
            cursor = connection.cursor()
            
            # Rechercher l'email dans toutes les tables
            tables_to_search = [
                'cleaned_docs',
                'donnees_assurance_s1_clients',
                'donnees_assurance_s1_contrats',
                'donnees_assurance_s1_sinistres',
                'donnees_assurance_s1_devis',
                'donnees_assurance_s1_mapping_produits'
            ]
            
            found_data = []
            
            for table in tables_to_search:
                try:
                    # Obtenir les colonnes de la table
                    cursor.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                    """)
                    columns = [row[0] for row in cursor.fetchall()]
                    
                    # Rechercher l'email dans toutes les colonnes texte
                    for column in columns:
                        if any(keyword in column.lower() for keyword in ['email', 'mail', 'contact']):
                            cursor.execute(f"""
                                SELECT * FROM "{table}" 
                                WHERE LOWER("{column}") = LOWER(%s)
                                LIMIT 10
                            """, (email,))
                            
                            rows = cursor.fetchall()
                            if rows:
                                for row in rows:
                                    found_data.append({
                                        'table': table,
                                        'column': column,
                                        'data': dict(zip(columns, row))
                                    })
                                    
                except Exception as table_error:
                    logger.info(f"   Table {table} ignorée: {table_error}")
                    continue
            
            connection.close()
            
            if found_data:
                logger.info(f"✅ Email {email} trouvé dans {len(found_data)} enregistrements")
                return True, found_data
            else:
                logger.info(f"❌ Email {email} non trouvé dans la base de données")
                return False, []
                
        except Exception as e:
            logger.error(f"❌ Erreur recherche email dans DB: {e}")
            return False, []
    
    def generate_verification_code(self) -> str:
        """Génère un code de vérification à 6 chiffres"""
        return ''.join(random.choices(string.digits, k=6))
    
    def send_verification_email(self, email: str, code: str) -> bool:
        """Envoie le code de vérification par email"""
        try:
            # Créer le message
            msg = MIMEMultipart()
            msg['From'] = f"{self.email_config['from_name']} <{self.email_config['email']}>"
            msg['To'] = email
            msg['Subject'] = "Code de vérification - Système Assurance RAG"
            
            # Corps du message
            body = f"""
            Bonjour,
            
            Vous avez demandé à accéder à vos données personnelles d'assurance.
            
            Votre code de vérification est : {code}
            
            Ce code est valide pendant 10 minutes.
            
            Si vous n'avez pas demandé cet accès, ignorez cet email.
            
            Cordialement,
            Système Assurance RAG
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connexion SMTP
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            
            # Envoi
            text = msg.as_string()
            server.sendmail(self.email_config['email'], email, text)
            server.quit()
            
            logger.info(f"✅ Code de vérification envoyé à {email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi email: {e}")
            return False
    
    def store_verification_code(self, email: str, code: str) -> None:
        """Stocke le code de vérification avec timestamp"""
        self.verification_codes[email] = {
            'code': code,
            'timestamp': datetime.now(),
            'attempts': 0
        }
    
    def verify_code(self, email: str, provided_code: str) -> bool:
        """Vérifie le code fourni par l'utilisateur"""
        if email not in self.verification_codes:
            return False
        
        stored_data = self.verification_codes[email]
        
        # Vérifier l'expiration (10 minutes)
        if datetime.now() - stored_data['timestamp'] > timedelta(minutes=10):
            del self.verification_codes[email]
            return False
        
        # Vérifier le nombre de tentatives (max 3)
        if stored_data['attempts'] >= 3:
            del self.verification_codes[email]
            return False
        
        # Vérifier le code
        if stored_data['code'] == provided_code:
            del self.verification_codes[email]
            return True
        else:
            stored_data['attempts'] += 1
            return False
    
    def get_personal_data(self, email: str) -> List[Dict]:
        """Récupère les données personnelles de l'utilisateur depuis PostgreSQL"""
        try:
            import psycopg2
            
            connection = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                dbname=self.postgres_config['dbname'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                client_encoding='utf8'
            )
            
            cursor = connection.cursor()
            
            # Rechercher toutes les données liées à cet email
            tables_to_search = [
                'cleaned_docs',
                'donnees_assurance_s1_clients',
                'donnees_assurance_s1_contrats',
                'donnees_assurance_s1_sinistres',
                'donnees_assurance_s1_devis',
                'donnees_assurance_s1_mapping_produits'
            ]
            
            personal_data = []
            
            for table in tables_to_search:
                try:
                    # Obtenir les colonnes de la table
                    cursor.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                    """)
                    columns = [row[0] for row in cursor.fetchall()]
                    
                    # Rechercher l'email dans toutes les colonnes
                    for column in columns:
                        if any(keyword in column.lower() for keyword in ['email', 'mail', 'contact', 'nom', 'prenom', 'client']):
                            cursor.execute(f"""
                                SELECT * FROM "{table}" 
                                WHERE LOWER("{column}") LIKE LOWER(%s)
                                LIMIT 50
                            """, (f'%{email}%',))
                            
                            rows = cursor.fetchall()
                            if rows:
                                for row in rows:
                                    # Formater les données
                                    formatted_data = {}
                                    for i, col in enumerate(columns):
                                        if row[i] is not None:
                                            formatted_data[col] = str(row[i])
                                    
                                    personal_data.append({
                                        'table': table,
                                        'column_found': column,
                                        'data': formatted_data,
                                        'created_at': datetime.now().isoformat()
                                    })
                                    
                except Exception as table_error:
                    logger.info(f"   Table {table} ignorée: {table_error}")
                    continue
            
            connection.close()
            
            logger.info(f"✅ {len(personal_data)} données personnelles récupérées pour {email}")
            return personal_data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données: {e}")
            return []
    
    def process_personal_data_request(self, query: str, user_email: str = None, verification_code: str = None) -> Dict:
        """Traite une demande de données personnelles"""
        
        # Étape 1: Détecter si c'est une demande de données personnelles
        if not self.detect_personal_data_request(query):
            return {
                'is_personal_data': False,
                'message': 'Cette requête ne concerne pas vos données personnelles.'
            }
        
        # Étape 2: Si pas d'email, demander l'email
        if not user_email:
            return {
                'is_personal_data': True,
                'needs_email': True,
                'message': 'Pour accéder à vos données personnelles, veuillez fournir votre adresse email.'
            }
        
        # Étape 3: Vérifier l'email dans la base de données
        email_exists, email_data = self.search_email_in_database(user_email)
        if not email_exists:
            return {
                'is_personal_data': True,
                'email_verified': False,
                'message': f'L\'email {user_email} n\'a pas été trouvé dans nos bases de données.'
            }
        
        # Étape 4: Si pas de code, envoyer le code
        if not verification_code:
            code = self.generate_verification_code()
            if self.send_verification_email(user_email, code):
                self.store_verification_code(user_email, code)
                return {
                    'is_personal_data': True,
                    'email_verified': True,
                    'needs_verification': True,
                    'message': f'Un code de vérification a été envoyé à {user_email}. Veuillez le saisir pour accéder à vos données.'
                }
            else:
                return {
                    'is_personal_data': True,
                    'email_verified': True,
                    'verification_error': True,
                    'message': 'Erreur lors de l\'envoi du code de vérification. Veuillez réessayer.'
                }
        
        # Étape 5: Vérifier le code
        if not self.verify_code(user_email, verification_code):
            return {
                'is_personal_data': True,
                'email_verified': True,
                'verification_failed': True,
                'message': 'Code de vérification incorrect ou expiré. Veuillez réessayer.'
            }
        
        # Étape 6: Récupérer et retourner les données
        personal_data = self.get_personal_data(user_email)
        return {
            'is_personal_data': True,
            'email_verified': True,
            'verification_success': True,
            'personal_data': personal_data,
            'message': f'Voici vos données personnelles ({len(personal_data)} éléments trouvés):'
        }

def main():
    """Test du système d'authentification"""
    auth_system = PersonalDataAuthSystem()
    
    # Test 1: Détection de demande personnelle
    test_queries = [
        "Qu'est-ce que l'assurance auto ?",
        "Je veux voir mon contrat",
        "Montrez-moi mes données",
        "Quelles sont mes garanties ?"
    ]
    
    for query in test_queries:
        is_personal = auth_system.detect_personal_data_request(query)
        print(f"Query: '{query}' -> Personal: {is_personal}")
    
    # Test 2: Processus complet
    result = auth_system.process_personal_data_request("Je veux voir mon contrat")
    print(f"\nRésultat: {result}")

if __name__ == "__main__":
    main()
