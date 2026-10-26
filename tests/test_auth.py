import time

from src.auth import client_auth
from src.auth.client_auth import MAX_CODE_ATTEMPTS, ClientAuthService

from .fakes import CLIENT, CapturingSender, FakeRepository


def make_service():
    sender = CapturingSender()
    return ClientAuthService(FakeRepository(), sender), sender


async def test_code_is_sent_to_the_registered_email_and_works_once():
    service, sender = make_service()
    await service.start("s1", "Ben Ali", "Mariem", "98765432")
    assert sender.sent[0][0] == CLIENT["email"]
    code = sender.last_code
    assert len(code) == 6 and code.isdigit()

    result = service.verify("s1", code)
    assert result.ok and result.token
    assert service.client_id_for(result.token) == CLIENT["id"]
    assert not service.verify("s1", code).ok  # single use


async def test_unknown_client_gets_no_email_and_cannot_verify():
    service, sender = make_service()
    await service.start("s1", "Inconnu", "Personne", "11111111")
    assert sender.sent == []
    assert not service.verify("s1", "000000").ok


async def test_code_is_bound_to_the_session_that_requested_it():
    service, sender = make_service()
    await service.start("victim", "Ben Ali", "Mariem", "98765432")
    assert not service.verify("attacker", sender.last_code).ok


async def test_code_locks_after_three_wrong_attempts():
    service, sender = make_service()
    await service.start("s1", "Ben Ali", "Mariem", "98765432")
    good = sender.last_code
    wrong = "000000" if good != "000000" else "111111"
    for expected_left in range(MAX_CODE_ATTEMPTS - 1, -1, -1):
        result = service.verify("s1", wrong)
        assert not result.ok and result.attempts_left == expected_left
    assert not service.verify("s1", good).ok  # locked, even with the right code


async def test_code_expires(monkeypatch):
    service, sender = make_service()
    await service.start("s1", "Ben Ali", "Mariem", "98765432")
    real = time.monotonic
    monkeypatch.setattr(client_auth.time, "monotonic", lambda: real() + client_auth.CODE_TTL_SECONDS + 1)
    result = service.verify("s1", sender.last_code)
    assert not result.ok and result.expired


async def test_identity_attempts_are_limited():
    service, _ = make_service()
    for _ in range(client_auth.MAX_IDENTITY_ATTEMPTS):
        await service.start("s1", "X", "Y", "11111111")
    assert service.identity_attempts_exceeded("s1")
    service.forget_session("s1")
    assert service.identity_attempts_exceeded("s1")  # cancelling does not reset the limit


async def test_codes_are_stored_hashed():
    service, sender = make_service()
    await service.start("s1", "Ben Ali", "Mariem", "98765432")
    stored = service._challenges["s1"]
    assert sender.last_code not in stored.code_hash
