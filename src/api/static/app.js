// BHbot chat page.
// Everything coming from the server is inserted with textContent, never innerHTML,
// so an answer or a document can't inject HTML or scripts into the page.

(function () {
  "use strict";

  const chatMessages = document.getElementById("chatMessages");
  const messageInput = document.getElementById("messageInput");
  const sendButton = document.getElementById("sendButton");
  const quotationModal = document.getElementById("quotationModal");
  const quotationForm = document.getElementById("quotationForm");
  const quotationResult = document.getElementById("quotationResult");
  const submitQuotationBtn = document.getElementById("submitQuotationBtn");

  let isTyping = false;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function addMessage(text, isUser) {
    const messageDiv = el("div", "message " + (isUser ? "user" : "assistant"));
    messageDiv.appendChild(el("div", "message-avatar", isUser ? "U" : "BH"));
    const contentDiv = el("div", "message-content");
    contentDiv.appendChild(el("div", "message-text", text));
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return contentDiv;
  }

  function addTypingIndicator() {
    const messageDiv = el("div", "message assistant");
    messageDiv.appendChild(el("div", "message-avatar", "BH"));
    const contentDiv = el("div", "message-content");
    const typing = el("div", "typing-indicator");
    typing.appendChild(el("span", null, "BHbot écrit"));
    const dots = el("div", "typing-dots");
    for (let i = 0; i < 3; i += 1) dots.appendChild(el("div", "typing-dot"));
    typing.appendChild(dots);
    contentDiv.appendChild(typing);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
  }

  function addAssistantMessage(data) {
    const contentDiv = addMessage(data.response, false);

    if (data.quotation) {
      const card = el("div", "quotation-card");
      card.appendChild(el("div", "quotation-title", "Devis généré"));
      card.appendChild(el("div", "quotation-content", JSON.stringify(data.quotation, null, 2)));
      contentDiv.appendChild(card);
    }

    if (Array.isArray(data.sources) && data.sources.length > 0) {
      const box = el("div", "message-sources", "Sources :");
      const list = el("ol");
      data.sources.forEach(function (source) {
        list.appendChild(el("li", null, source.title + " (" + source.kind + ")"));
      });
      box.appendChild(list);
      contentDiv.appendChild(box);
    }
    scrollToBottom();
  }

  async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isTyping) return;

    addMessage(message, true);
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendButton.disabled = true;
    isTyping = true;
    const typingDiv = addTypingIndicator();

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message }),
      });
      const result = await response.json();
      typingDiv.remove();
      if (response.ok && result.response) {
        addAssistantMessage(result);
      } else {
        addMessage("Désolé, je n'ai pas pu traiter votre demande.", false);
      }
    } catch (error) {
      console.error("Erreur:", error);
      typingDiv.remove();
      addMessage("Désolé, une erreur s'est produite. Veuillez réessayer.", false);
    } finally {
      sendButton.disabled = false;
      isTyping = false;
      messageInput.focus();
    }
  }

  async function clearChat() {
    if (!window.confirm("Êtes-vous sûr de vouloir effacer toute la conversation ?")) return;
    try {
      await fetch("/api/session/reset", { method: "POST" });
    } catch (error) {
      console.error("Erreur:", error);
    }
    chatMessages.replaceChildren();
    const welcome = el("div", "welcome-message");
    welcome.appendChild(el("div", "welcome-title", "Bonjour ! Je suis BHbot"));
    welcome.appendChild(el("div", "example-questions"));
    chatMessages.appendChild(welcome);
  }

  function openQuotationModal() {
    quotationModal.style.display = "flex";
    quotationModal.classList.add("show");
    document.body.style.overflow = "hidden";
  }

  function closeQuotationModal() {
    quotationModal.style.display = "none";
    quotationModal.classList.remove("show");
    document.body.style.overflow = "auto";
    quotationResult.style.display = "none";
  }

  function showQuotationResult(ok, title, details) {
    quotationResult.style.display = "block";
    quotationResult.className = "quotation-result " + (ok ? "success" : "error");
    quotationResult.replaceChildren(el("h3", null, title));
    if (details) quotationResult.appendChild(el("pre", null, details));
  }

  async function handleQuotationSubmit(event) {
    event.preventDefault();
    const formData = new FormData(quotationForm);
    const toInt = function (name, fallback) {
      const value = formData.get(name);
      return value === null || value === "" ? fallback : parseInt(value, 10);
    };
    const payload = {
      n_cin: formData.get("n_cin"),
      valeur_venale: toInt("valeur_venale"),
      nature_contrat: formData.get("nature_contrat"),
      nombre_place: toInt("nombre_place"),
      valeur_a_neuf: toInt("valeur_a_neuf"),
      date_premiere_mise_en_circulation: formData.get("date_premiere_mise_en_circulation"),
      capital_bris_de_glace: toInt("capital_bris_de_glace", 900),
      capital_dommage_collision: toInt("capital_dommage_collision", 60000),
      puissance: toInt("puissance"),
      classe: toInt("classe"),
    };

    submitQuotationBtn.disabled = true;
    submitQuotationBtn.textContent = "Génération en cours...";
    try {
      const response = await fetch("/api/quotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok || result.error) {
        const message = result.message || (result.detail ? JSON.stringify(result.detail, null, 2) : "Erreur inconnue");
        showQuotationResult(false, "Erreur lors de la génération", message);
      } else {
        showQuotationResult(true, "Devis généré avec succès", JSON.stringify(result.quotation, null, 2));
        addMessage("Devis généré avec succès. Détails dans la fenêtre du devis.", false);
      }
    } catch (error) {
      console.error("Erreur devis:", error);
      showQuotationResult(false, "Erreur de connexion", "Impossible de joindre le service de devis.");
    } finally {
      submitQuotationBtn.disabled = false;
      submitQuotationBtn.textContent = "Générer le Devis";
    }
  }

  messageInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });
  messageInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  sendButton.addEventListener("click", sendMessage);
  document.getElementById("devisAutoBtn").addEventListener("click", openQuotationModal);
  document.querySelector(".clear-button").addEventListener("click", clearChat);
  document.querySelectorAll('[data-action="close-quotation"]').forEach(function (button) {
    button.addEventListener("click", closeQuotationModal);
  });
  quotationForm.addEventListener("submit", handleQuotationSubmit);
  messageInput.focus();
})();
