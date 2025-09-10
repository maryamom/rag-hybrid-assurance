#!/usr/bin/env python3
"""
API FastAPI pour le RAG Hybride avec LangChain
Interface web pour interroger le système RAG hybride orchestré par LangChain
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime
import json

from hybrid_rag_langchain_orchestrator import HybridRAGLangChainOrchestrator

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic
class QueryRequest(BaseModel):
    query: str
    include_quotation: Optional[bool] = True
    clear_memory: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    response: str
    vector_results_count: int
    graph_results_count: int
    fused_results_count: int
    quotation: Optional[Dict[str, Any]] = None
    needs_quotation: bool
    timestamp: str
    processing_time: Optional[float] = None
    sources: Optional[List[Dict[str, Any]]] = None

class MemoryRequest(BaseModel):
    action: str  # 'clear' ou 'get'

class QuotationRequest(BaseModel):
    n_cin: str
    valeur_venale: int
    nature_contrat: str
    nombre_place: int
    valeur_a_neuf: int
    date_premiere_mise_en_circulation: str
    capital_bris_de_glace: Optional[int] = 900
    capital_dommage_collision: Optional[int] = 60000
    puissance: int
    classe: int

class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    neo4j: bool
    postgres: bool
    openrouter: bool
    langchain: bool
    timestamp: str

# Initialisation de l'application
app = FastAPI(
    title="RAG Hybride Assurance - LangChain",
    description="Système RAG hybride avec LangChain combinant recherche vectorielle et graphe pour l'assurance",
    version="2.0.0"
)

# Instance globale de l'orchestrateur
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global orchestrator
    try:
        orchestrator = HybridRAGLangChainOrchestrator()
        await orchestrator.initialize()
        logger.info("🚀 RAG Hybride LangChain initialisé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur d'initialisation: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    global orchestrator
    if orchestrator:
        logger.info("🔌 Connexions fermées")

@app.get("/", response_class=HTMLResponse)
async def root():

    """Interface de conversation style ChatGPT"""
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BHbot - Assistant Assurance IA</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                background: #0f0f0f;
                color: #ffffff;
                height: 100vh;
                overflow: hidden;
            }
            
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 100vh;
                max-width: 100%;
                margin: 0 auto;
            }
            
            .chat-header {
                background: #212121;
                padding: 1rem 1.5rem;
                border-bottom: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-shrink: 0;
            }
            
            .chat-title {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .chat-title h1 {
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
            }
            
            .chat-subtitle {
                font-size: 0.875rem;
                color: #9ca3af;
                margin-top: 0.25rem;
            }
            
            .chat-actions {
                display: flex;
                gap: 0.5rem;
                align-items: center;
            }
            
            .btn-quotation {
                background: #10b981;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-size: 0.875rem;
                font-weight: 500;
                transition: background-color 0.2s;
            }
            
            .btn-quotation:hover {
                background: #059669;
            }
            
            .quotation-modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                z-index: 1000;
                overflow-y: auto;
            }
            
            .quotation-modal.show {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }
            
            .quotation-content {
                background: #1f1f1f;
                border-radius: 1rem;
                padding: 2rem;
                max-width: 800px;
                width: 100%;
                max-height: 90vh;
                overflow-y: auto;
                border: 1px solid #333;
            }
            
            .quotation-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
            }
            
            .quotation-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #ffffff;
            }
            
            .btn-close {
                background: none;
                border: none;
                color: #9ca3af;
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0.5rem;
                border-radius: 0.5rem;
                transition: background-color 0.2s;
            }
            
            .btn-close:hover {
                background: #333;
            }
            
            .quotation-form {
                display: grid;
                gap: 1.5rem;
            }
            
            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }
            
            .form-group {
                display: flex;
                flex-direction: column;
            }
            
            .form-group label {
                color: #ffffff;
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            
            .form-group input,
            .form-group select {
                background: #333;
                border: 1px solid #555;
                color: #ffffff;
                padding: 0.75rem;
                border-radius: 0.5rem;
                font-size: 0.875rem;
            }
            
            .form-group input:focus,
            .form-group select:focus {
                outline: none;
                border-color: #10b981;
            }
            
            .form-group .help-text {
                font-size: 0.75rem;
                color: #9ca3af;
                margin-top: 0.25rem;
            }
            
            .required {
                color: #ef4444;
            }
            
            .quotation-result {
                margin-top: 1.5rem;
                padding: 1rem;
                border-radius: 0.5rem;
                display: none;
            }
            
            .quotation-result.success {
                background: #064e3b;
                border: 1px solid #10b981;
                color: #10b981;
            }
            
            .quotation-result.error {
                background: #7f1d1d;
                border: 1px solid #ef4444;
                color: #ef4444;
            }
            
            .quotation-actions {
                display: flex;
                gap: 1rem;
                margin-top: 2rem;
            }
            
            .btn-primary {
                background: #10b981;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-weight: 500;
                flex: 1;
            }
            
            .btn-primary:hover {
                background: #059669;
            }
            
            .btn-primary:disabled {
                background: #6b7280;
                cursor: not-allowed;
            }
            
            .btn-secondary {
                background: #374151;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-weight: 500;
            }
            
            .btn-secondary:hover {
                background: #4b5563;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.875rem;
                color: #10b981;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                background: #10b981;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                scroll-behavior: smooth;
            }
            
            .message {
                display: flex;
                gap: 0.75rem;
                max-width: 80%;
                animation: fadeInUp 0.3s ease-out;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                align-self: flex-end;
                flex-direction: row-reverse;
            }
            
            .message.assistant {
                align-self: flex-start;
            }
            
            .message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.875rem;
                font-weight: 600;
                flex-shrink: 0;
            }
            
            .message.user .message-avatar {
                background: #3b82f6;
                color: white;
            }
            
            .message.assistant .message-avatar {
                background: #10b981;
                color: white;
            }
            
            .message-content {
                background: #1f1f1f;
                padding: 0.75rem 1rem;
                border-radius: 1rem;
                border: 1px solid #333;
                position: relative;
                word-wrap: break-word;
            }
            
            .message.user .message-content {
                background: #3b82f6;
                border-color: #3b82f6;
            }
            
            .message-text {
                line-height: 1.5;
                color: #ffffff;
            }
            
            .message.user .message-text {
                color: #ffffff;
            }
            
            .quotation-card {
                background: #0f172a;
                border: 1px solid #1e40af;
                border-radius: 0.75rem;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            
            .quotation-title {
                font-size: 0.875rem;
                font-weight: 600;
                color: #60a5fa;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .quotation-content {
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.75rem;
                color: #cbd5e1;
                background: #1e293b;
                padding: 0.75rem;
                border-radius: 0.5rem;
                overflow-x: auto;
            }
            
            .sources-section {
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid #333;
            }
            
            .sources-title {
                font-size: 0.75rem;
                color: #9ca3af;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .source-item {
                background: #1f1f1f;
                border: 1px solid #333;
                border-radius: 0.5rem;
                padding: 0.5rem 0.75rem;
                margin: 0.25rem 0;
                font-size: 0.75rem;
                color: #9ca3af;
            }
            
            .stats {
                display: flex;
                gap: 1rem;
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid #333;
            }
            
            .stat {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-size: 0.75rem;
            }
            
            .stat-number {
                font-weight: 600;
                color: #60a5fa;
            }
            
            .stat-label {
                color: #9ca3af;
            }
            
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #9ca3af;
                font-style: italic;
            }
            
            .typing-dots {
                display: flex;
                gap: 0.25rem;
            }
            
            .typing-dot {
                width: 4px;
                height: 4px;
                background: #9ca3af;
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
            
            .chat-input-container {
                background: #212121;
                padding: 1rem 1.5rem;
                border-top: 1px solid #333;
                flex-shrink: 0;
            }
            
            .chat-input-wrapper {
                display: flex;
                align-items: flex-end;
                gap: 0.75rem;
                background: #1f1f1f;
                border: 1px solid #333;
                border-radius: 1.5rem;
                padding: 0.75rem 1rem;
                transition: border-color 0.2s;
            }
            
            .chat-input-wrapper:focus-within {
                border-color: #3b82f6;
            }
            
            .chat-input {
                flex: 1;
                background: transparent;
                border: none;
                outline: none;
                color: #ffffff;
                font-size: 1rem;
                line-height: 1.5;
                resize: none;
                max-height: 120px;
                min-height: 24px;
                font-family: inherit;
            }
            
            .chat-input::placeholder {
                color: #9ca3af;
            }
            
            .send-button {
                background: #3b82f6;
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s;
                flex-shrink: 0;
            }
            
            .send-button:hover:not(:disabled) {
                background: #2563eb;
                transform: scale(1.05);
            }
            
            .send-button:disabled {
                background: #374151;
                cursor: not-allowed;
            }
            
            .send-icon {
                width: 16px;
                height: 16px;
                fill: white;
            }
            
            .welcome-message {
                text-align: center;
                padding: 2rem;
                color: #9ca3af;
            }
            
            .welcome-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #ffffff;
                margin-bottom: 0.5rem;
            }
            
            .welcome-subtitle {
                font-size: 1rem;
                margin-bottom: 2rem;
            }
            
            .example-questions {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 0.75rem;
                max-width: 600px;
                margin: 0 auto;
            }
            
            .example-question {
                background: #1f1f1f;
                border: 1px solid #333;
                border-radius: 0.75rem;
                padding: 0.75rem 1rem;
                cursor: pointer;
                transition: all 0.2s;
                text-align: left;
                font-size: 0.875rem;
            }
            
            .example-question:hover {
                background: #2a2a2a;
                border-color: #3b82f6;
            }
            
            .clear-button {
                background: transparent;
                border: 1px solid #374151;
                color: #9ca3af;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-size: 0.875rem;
                transition: all 0.2s;
            }
            
            .clear-button:hover {
                background: #374151;
                color: #ffffff;
            }
            
            .error-message {
                background: #7f1d1d;
                border: 1px solid #dc2626;
                color: #fecaca;
                padding: 0.75rem 1rem;
                border-radius: 0.75rem;
                margin: 0.5rem 0;
            }
            
            @media (max-width: 768px) {
                .message {
                    max-width: 95%;
                }
                
                .chat-header {
                    padding: 1rem;
                }
                
                .chat-input-container {
                    padding: 1rem;
                }
                
                .example-questions {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-title">
                    <div class="message-avatar" style="background: #10b981;">BH</div>
                    <div>
                        <h1>BHbot</h1>
                        <div class="chat-subtitle">Assistant Assurance IA</div>
                    </div>
                </div>
                <div class="chat-actions">
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span>En ligne</span>
                    </div>
                </div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <div class="welcome-title">Bonjour ! Je suis BHbot</div>
                    <div class="example-questions">
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <div class="chat-input-wrapper">
                    <textarea 
                        id="messageInput" 
                        class="chat-input" 
                        placeholder="Tapez votre message..."
                        rows="1"
                    ></textarea>
                    <button id="sendButton" class="send-button">
                        <svg class="send-icon" viewBox="0 0 24 24">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="color: #9ca3af; font-size: 0.875rem;">
                            Pour avoir un devis auto cliquez ici
                        </div>
                        <button id="devisAutoBtn" style="background: #10b981; color: white; border: 1px solid #10b981; font-size: 0.8rem; padding: 0.4rem 0.8rem; border-radius: 0.375rem; cursor: pointer;">
                            Devis Auto
                        </button>
                    </div>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <button class="clear-button">Effacer la conversation</button>
                        <div style="font-size: 0.75rem; color: #9ca3af;">
                            Appuyez sur Entrée pour envoyer
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Modal de Devis -->
        <div id="quotationModal" class="quotation-modal">
            <div class="quotation-content">
                <div class="quotation-header">
                    <h2 class="quotation-title">Devis d'Assurance Auto</h2>
                    <button class="btn-close" onclick="closeQuotationModal()">&times;</button>
                </div>
                
                <form id="quotationForm" class="quotation-form">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="n_cin">CIN <span class="required">*</span></label>
                            <input type="text" id="n_cin" name="n_cin" placeholder="08478931" required>
                            <div class="help-text">8 chiffres</div>
                        </div>
                        
                        <div class="form-group">
                            <label for="valeur_venale">Valeur Vénale (€) <span class="required">*</span></label>
                            <input type="number" id="valeur_venale" name="valeur_venale" placeholder="60000" required>
                            <div class="help-text">Valeur actuelle du véhicule</div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="nature_contrat">Type de Contrat <span class="required">*</span></label>
                            <select id="nature_contrat" name="nature_contrat" required>
                                <option value="r">Responsabilité Civile</option>
                                <option value="n">Tous Risques</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="nombre_place">Nombre de Places <span class="required">*</span></label>
                            <input type="number" id="nombre_place" name="nombre_place" placeholder="5" min="2" max="9" required>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="valeur_a_neuf">Valeur à Neuf (€) <span class="required">*</span></label>
                            <input type="number" id="valeur_a_neuf" name="valeur_a_neuf" placeholder="60000" required>
                            <div class="help-text">Valeur d'achat du véhicule</div>
                        </div>
                        
                        <div class="form-group">
                            <label for="date_premiere_mise_en_circulation">Date de Mise en Circulation <span class="required">*</span></label>
                            <input type="date" id="date_premiere_mise_en_circulation" name="date_premiere_mise_en_circulation" required>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="capital_bris_de_glace">Capital Bris de Glace (€)</label>
                            <input type="number" id="capital_bris_de_glace" name="capital_bris_de_glace" placeholder="900">
                        </div>
                        
                        <div class="form-group">
                            <label for="capital_dommage_collision">Capital Dommage Collision (€)</label>
                            <input type="number" id="capital_dommage_collision" name="capital_dommage_collision" placeholder="60000">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="puissance">Puissance Fiscale <span class="required">*</span></label>
                            <input type="number" id="puissance" name="puissance" placeholder="6" min="1" max="20" required>
                            <div class="help-text">CV fiscaux</div>
                        </div>
                        
                        <div class="form-group">
                            <label for="classe">Classe Bonus/Malus <span class="required">*</span></label>
                            <input type="number" id="classe" name="classe" placeholder="3" min="1" max="50" required>
                            <div class="help-text">Classe de bonus/malus</div>
                        </div>
                    </div>
                    
                    <div class="quotation-result" id="quotationResult"></div>
                    
                    <div class="quotation-actions">
                        <button type="button" class="btn-secondary" onclick="closeQuotationModal()">
                            Annuler
                        </button>
                        <button type="submit" class="btn-primary" id="submitQuotationBtn">
                            Générer le Devis
                        </button>
                    </div>
                </form>
            </div>
        </div>
        <script>
            let isTyping = false;
            
            function addMessage(text, isUser = false) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
                
                const avatarDiv = document.createElement('div');
                avatarDiv.className = 'message-avatar';
                avatarDiv.textContent = isUser ? 'U' : 'BH';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                const textDiv = document.createElement('div');
                textDiv.className = 'message-text';
                textDiv.textContent = text;
                
                contentDiv.appendChild(textDiv);
                messageDiv.appendChild(avatarDiv);
                messageDiv.appendChild(contentDiv);
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                return messageDiv;
            }
            
            async function sendMessage() {
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const message = messageInput.value.trim();
                
                if (!message || isTyping) return;
                
                // Ajouter le message utilisateur
                addMessage(message, true);
                
                // Réinitialiser l'input et désactiver le bouton
                messageInput.value = '';
                sendButton.disabled = true;
                isTyping = true;
                
                // Afficher l'indicateur de frappe
                const typingDiv = addTypingIndicator();
                
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: message,
                            include_quotation: true
                        })
                    });
                    
                    const result = await response.json();
                    
                    // Supprimer l'indicateur de frappe
                    typingDiv.remove();
                    
                    if (result.response) {
                        addAssistantMessage(result);
                    } else {
                        addMessage("Désolé, je n'ai pas pu traiter votre demande.", false);
                    }
                    
                } catch (error) {
                    console.error('Erreur:', error);
                    typingDiv.remove();
                    addMessage("Désolé, une erreur s'est produite. Veuillez réessayer.", false);
                } finally {
                    sendButton.disabled = false;
                    isTyping = false;
                    messageInput.focus();
                }
            }
            
            function addTypingIndicator() {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                
                messageDiv.innerHTML = `
                    <div class="message-avatar">BH</div>
                    <div class="message-content">
                        <div class="typing-indicator">
                            <span>BHbot écrit</span>
                            <div class="typing-dots">
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                            </div>
                        </div>
                    </div>
                `;
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                return messageDiv;
            }
            
            function addAssistantMessage(data) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                
                let content = `
                    <div class="message-avatar">BH</div>
                    <div class="message-content">
                        <div class="message-text">${data.response.replace(/\n/g, '<br>')}</div>
                `;
                
                if (data.quotation) {
                    content += `
                        <div class="quotation-card">
                            <div class="quotation-title">Devis généré</div>
                            <div class="quotation-content">${JSON.stringify(data.quotation, null, 2)}</div>
                        </div>
                    `;
                }
                
                content += `</div></div>`;
                messageDiv.innerHTML = content;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function clearChat() {
                if (confirm('Êtes-vous sûr de vouloir effacer toute la conversation ?')) {
                    const chatMessages = document.getElementById('chatMessages');
                    chatMessages.innerHTML = '';
                    const welcomeDiv = document.createElement('div');
                    welcomeDiv.className = 'welcome-message';
                    welcomeDiv.innerHTML = '<div class="welcome-title">Bonjour ! Je suis BHbot</div><div class="example-questions"></div>';
                    chatMessages.appendChild(welcomeDiv);
                }
            }
            
            function openQuotationModal() {
                const modal = document.getElementById('quotationModal');
                if (modal) {
                    modal.style.display = 'flex';
                    modal.classList.add('show');
                    document.body.style.overflow = 'hidden';
                }
            }
            
            function closeQuotationModal() {
                const modal = document.getElementById('quotationModal');
                if (modal) {
                    modal.style.display = 'none';
                    modal.classList.remove('show');
                    document.body.style.overflow = 'auto';
                }
                const result = document.getElementById('quotationResult');
                if (result) {
                    result.style.display = 'none';
                }
            }
            
            async function handleQuotationSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = document.getElementById('submitQuotationBtn');
    const resultDiv = document.getElementById('quotationResult');
    
    // Récupérer les données du formulaire
    const formData = new FormData(form);
    const quotationData = {
        n_cin: formData.get('n_cin'),
        valeur_venale: parseInt(formData.get('valeur_venale')),
        nature_contrat: formData.get('nature_contrat'),
        nombre_place: parseInt(formData.get('nombre_place')),
        valeur_a_neuf: parseInt(formData.get('valeur_a_neuf')),
        date_premiere_mise_en_circulation: formData.get('date_premiere_mise_en_circulation'),
        capital_bris_de_glace: parseInt(formData.get('capital_bris_de_glace') || '900'),
        capital_dommage_collision: parseInt(formData.get('capital_dommage_collision') || '60000'),
        puissance: parseInt(formData.get('puissance')),
        classe: parseInt(formData.get('classe'))
    };
    
    // Désactiver le bouton de soumission
    submitBtn.disabled = true;
    submitBtn.textContent = 'Génération en cours...';
    
    try {
        const response = await fetch('/api/quotation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(quotationData)
        });
        
        const result = await response.json();
        
        // Afficher le résultat
        resultDiv.style.display = 'block';
        
        if (result.error) {
            resultDiv.className = 'quotation-result error';
            resultDiv.innerHTML = `
                <h3>❌ Erreur lors de la génération</h3>
                <p>${result.message}</p>
            `;
        } else {
            resultDiv.className = 'quotation-result success';
            const prime = result.prime || result.montant || 'Non spécifié';
            resultDiv.innerHTML = `
                <h3>✅ Devis généré avec succès !</h3>
                <p>Prime annuelle: ${prime} DT</p>
                <pre>${JSON.stringify(result, null, 2)}</pre>
            `;
            
            // Ajouter le devis à la conversation
            addMessage(`Devis généré avec succès ! Prime annuelle: ${prime} DT`, false);
            
            // Fermer le modal après 3 secondes
            setTimeout(closeQuotationModal, 3000);
        }
        
    } catch (error) {
        console.error('Erreur devis:', error);
        resultDiv.className = 'quotation-result error';
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `
            <h3>❌ Erreur de connexion</h3>
            <p>Impossible de se connecter au service de devis</p>
        `;
    } finally {
        // Réactiver le bouton
        submitBtn.disabled = false;
        submitBtn.textContent = 'Générer le Devis';
    }
}
                
         
            // Auto-resize textarea
            document.getElementById('messageInput').addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            });
            
            // Envoyer message avec Entrée
            document.getElementById('messageInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // Bouton Envoyer
            document.getElementById('sendButton').onclick = sendMessage;
            
            // Bouton Devis Auto
            document.getElementById('devisAutoBtn').onclick = openQuotationModal;
            
            // Bouton Effacer
            document.querySelector('.clear-button').onclick = clearChat;
            
            // Formulaire de devis
            document.getElementById('quotationForm').onsubmit = handleQuotationSubmit;
            
            // Focus sur l'input
            document.getElementById('messageInput').focus();
        </script>
        
    </body>
    </html>
    """

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Traite une requête utilisateur avec LangChain"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service non disponible")
    
    try:
        start_time = datetime.now()
        
        # Effacer la mémoire si demandé
        if request.clear_memory:
            orchestrator.clear_memory()
        
        result = await orchestrator.process_query(request.query)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        result['processing_time'] = processing_time
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Erreur traitement requête: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory")
async def manage_memory(request: MemoryRequest):
    """Gère la mémoire de conversation"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service non disponible")
    
    try:
        if request.action == "clear":
            orchestrator.clear_memory()
            return {"success": True, "message": "Mémoire effacée"}
        elif request.action == "get":
            # Retourner l'état de la mémoire
            memory_state = {
                "messages_count": len(orchestrator.memory.chat_memory.messages),
                "messages": [
                    {"type": msg.__class__.__name__, "content": msg.content}
                    for msg in orchestrator.memory.chat_memory.messages
                ]
            }
            return {"success": True, "memory": memory_state}
        else:
            raise HTTPException(status_code=400, detail="Action non supportée")
            
    except Exception as e:
        logger.error(f"Erreur gestion mémoire: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état des services"""
    if not orchestrator:
        return HealthResponse(
            status="error",
            qdrant=False,
            neo4j=False,
            postgres=False,
            openrouter=False,
            langchain=False,
            timestamp=datetime.now().isoformat()
        )
    
    try:
        # Vérifications basiques
        qdrant_ok = orchestrator.vector_retriever is not None
        neo4j_ok = orchestrator.graph_retriever is not None
        postgres_ok = True  # Assumé OK si l'orchestrateur est initialisé
        openrouter_ok = orchestrator.llm is not None
        langchain_ok = orchestrator.hybrid_retriever is not None
        
        status = "healthy" if all([qdrant_ok, neo4j_ok, postgres_ok, langchain_ok]) else "degraded"
        
        return HealthResponse(
            status=status,
            qdrant=qdrant_ok,
            neo4j=neo4j_ok,
            postgres=postgres_ok,
            openrouter=openrouter_ok,
            langchain=langchain_ok,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return HealthResponse(
            status="error",
            qdrant=False,
            neo4j=False,
            postgres=False,
            openrouter=False,
            langchain=False,
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/stats")
async def get_stats():
    """Retourne les statistiques du système"""
    try:
        # Statistiques basiques
        stats = {
            "timestamp": datetime.now().isoformat(),
            "services": {
                "qdrant": orchestrator.vector_retriever is not None,
                "neo4j": orchestrator.graph_retriever is not None,
                "postgres": True,
                "openrouter": orchestrator.llm is not None,
                "langchain": orchestrator.hybrid_retriever is not None
            },
            "memory": {
                "messages_count": len(orchestrator.memory.chat_memory.messages) if orchestrator.memory else 0
            },
            "config": {
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "deepseek/deepseek-chat-v3-0324:free",
                "vector_weight": 0.6,
                "graph_weight": 0.4
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quotation", response_class=HTMLResponse)
async def quotation_form():
    """Affiche le formulaire de devis"""
    try:
        with open("quotation_form.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Formulaire de devis non trouvé</h1>", status_code=404)

@app.post("/api/quotation")
async def generate_quotation(request: QuotationRequest):
    """Génère un devis via l'API externe"""
    try:
        # Convertir la requête en dictionnaire
        vehicle_info = request.dict()
        
        # Valider les informations
        from hybrid_rag_langchain_orchestrator import QuotationGenerator
        quotation_generator = QuotationGenerator({})
        validation = quotation_generator._validate_vehicle_info(vehicle_info)
        
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
        
        # Appel à l'API de devis
        import httpx
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
        return {
            'error': True,
            'message': f"Erreur lors de la génération du devis: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
