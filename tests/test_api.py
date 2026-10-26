"""End-to-end tests of the HTTP API with fake back-ends."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.api.main import SESSION_COOKIE, app

from .fakes import make_orchestrator


@pytest.fixture
def setup():
    orchestrator, sender = make_orchestrator()
    app.state.orchestrator = orchestrator
    yield orchestrator, sender
    del app.state.orchestrator


def browser():
    return TestClient(app)


def ask(client, text):
    response = client.post("/api/query", json={"query": text})
    assert response.status_code == 200
    return response.json()


def test_each_browser_gets_its_own_session_cookie(setup):
    a, b = browser(), browser()
    ask(a, "Qu'est-ce que le bris de glace ?")
    ask(b, "Qu'est-ce que le bris de glace ?")
    assert a.cookies.get(SESSION_COOKIE) != b.cookies.get(SESSION_COOKIE)


def test_general_question_without_llm_lists_sources(setup):
    result = ask(browser(), "Qu'est-ce que la garantie bris de glace ?")
    assert result["intent"] == "general"
    assert "pas configuré" in result["response"]
    assert result["sources"][0]["title"] == "Bris de glace"
    assert result["vector_results_count"] == 1 and result["graph_results_count"] == 1


def test_full_authentication_flow(setup):
    _, sender = setup
    client = browser()
    first = ask(client, "Je veux voir mes contrats")
    assert first["needs_auth"] and first["auth_step"] == "identity"

    second = ask(client, "Mariem Ben Ali 98 765 432")
    assert second["auth_step"] == "code"
    assert "example.com" not in second["response"]  # the e-mail address is never shown

    third = ask(client, sender.last_code)
    assert third["intent"] == "personal_data"
    assert "Identité vérifiée" in third["response"] and "Auto" in third["response"]

    follow_up = ask(client, "et mes contrats ?")  # already authenticated: no new code
    assert follow_up["intent"] == "personal_data" and not follow_up["needs_auth"]


def test_unknown_client_gets_the_same_answer_as_a_known_one(setup):
    known, unknown = browser(), browser()
    ask(known, "mes contrats")
    ask(unknown, "mes contrats")
    known_reply = ask(known, "Mariem Ben Ali 98765432")["response"]
    unknown_reply = ask(unknown, "Jean Inconnu 11111111")["response"]
    assert known_reply == unknown_reply


def test_another_browser_cannot_answer_someone_elses_code(setup):
    _, sender = setup
    victim, attacker = browser(), browser()
    ask(victim, "mes contrats")
    ask(victim, "Mariem Ben Ali 98765432")
    result = ask(attacker, sender.last_code)  # attacker has no authentication in progress
    assert result["intent"] == "general"
    assert "Identité vérifiée" not in result["response"]


def test_wrong_codes_then_lock(setup):
    client = browser()
    ask(client, "mes contrats")
    ask(client, "Mariem Ben Ali 98765432")
    assert "2 essai" in ask(client, "000001")["response"]
    assert "1 essai" in ask(client, "000002")["response"]
    assert "dépassé" in ask(client, "000003")["response"]


def test_cancel_authentication(setup):
    client = browser()
    ask(client, "mes contrats")
    result = ask(client, "annuler")
    assert result["intent"] == "general" and not result["needs_auth"]


def test_quote_conversation_collects_missing_fields(setup):
    orchestrator, _ = setup
    client = browser()
    first = ask(client, "Je veux un devis pour ma voiture de 6 CV, 5 places")
    assert first["intent"] == "quotation" and "CIN" in first["response"]
    second = ask(client, "CIN 12345678, valeur vénale 45000, valeur à neuf 52000, mise en circulation 2019, classe 3")
    assert second["quotation"] == {"prime": 812.5, "pack": "Confort"}
    assert orchestrator.quotation_client.calls[0].puissance == 6


def test_cin_is_masked_in_history(setup):
    orchestrator, _ = setup
    client = browser()
    ask(client, "Le bris de glace est-il inclus ? Mon CIN est 12345678")
    session_id = client.cookies.get(SESSION_COOKIE)
    assert all("12345678" not in m.content for m in orchestrator.sessions.get(session_id).history)


def test_reset_signs_out(setup):
    _, sender = setup
    client = browser()
    ask(client, "mes contrats")
    ask(client, "Mariem Ben Ali 98765432")
    ask(client, sender.last_code)
    client.post("/api/session/reset")
    assert ask(client, "mes contrats")["needs_auth"]


def test_old_memory_endpoint_is_gone(setup):
    response = browser().post("/api/memory", json={"action": "get"})
    assert response.status_code in (404, 405)


def test_quote_form_validation(setup):
    response = browser().post("/api/quotation", json={"n_cin": "12"})
    assert response.status_code == 422


def test_index_page_and_script_are_served(setup):
    client = browser()
    assert "app.js" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_llm_answer_uses_context_and_history(setup):
    llm = FakeListChatModel(responses=["Le bris de glace couvre le pare-brise [1].", "bris de glace franchise", "La franchise est indiquée au contrat [1]."])
    orchestrator, _ = make_orchestrator(llm=llm)
    app.state.orchestrator = orchestrator
    client = browser()
    assert "[1]" in ask(client, "Que couvre le bris de glace ?")["response"]
    ask(client, "Et la franchise ?")
    # the second question was rephrased with the history before retrieval
    assert orchestrator.retriever.queries[-1] == "bris de glace franchise"


def test_forged_session_id_is_replaced(setup):
    forged = "A" * 32
    response = browser().post(
        "/api/query", json={"query": "Qu'est-ce que le bris de glace ?"}, headers={"Cookie": f"{SESSION_COOKIE}={forged}"}
    )
    issued = response.cookies.get(SESSION_COOKIE)
    assert issued and issued != forged


def test_session_survives_between_requests(setup):
    client = browser()
    ask(client, "mes contrats")
    first = client.cookies.get(SESSION_COOKIE)
    assert ask(client, "Mariem Ben Ali 98765432")["auth_step"] == "code"  # same session: flow continues
    assert client.cookies.get(SESSION_COOKIE) == first
