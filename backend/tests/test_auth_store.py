import auth_store


def test_account_history_is_private_and_deletable(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "DB_PATH", tmp_path / "bloom.db")
    auth_store.initialise()
    first = auth_store.create_user("first@example.com", "long-password", "First school?", "North")
    second = auth_store.create_user("second@example.com", "long-password", "First school?", "South")

    auth_store.save_history(first["id"], {"concerns": {"acne": {"severity": "Mild"}}})
    records = auth_store.get_history(first["id"])
    assert len(records) == 1
    assert auth_store.get_history(second["id"]) == []
    assert auth_store.delete_history_entry(second["id"], records[0]["id"]) is False
    assert auth_store.delete_history_entry(first["id"], records[0]["id"]) is True
    assert auth_store.get_history(first["id"]) == []


def test_password_and_token_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "DB_PATH", tmp_path / "bloom.db")
    auth_store.initialise()
    auth_store.create_user("person@example.com", "long-password", "First school?", "North")
    assert auth_store.authenticate("person@example.com", "wrong") is None
    user = auth_store.authenticate("person@example.com", "long-password")
    assert user and auth_store.user_from_token(auth_store.issue_token(user))["id"] == user["id"]
    assert auth_store.security_question("person@example.com") == "First school?"
    assert not auth_store.reset_password("person@example.com", "wrong", "new-long-password")
    assert auth_store.reset_password("person@example.com", "north", "new-long-password")
    assert auth_store.authenticate("person@example.com", "new-long-password")
