"""Small, dependency-free account and private scan-history store.

This is deliberately kept separate from the image pipeline.  Passwords and
security answers are salted PBKDF2-HMAC hashes; images are never persisted.
Tokens are signed, short-lived bearer tokens suitable for this single-service
deployment. A configured AUTH_SECRET is preferred; local deployments persist a
generated key beside the database.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("BLOOM_DB_PATH", "data/bloom.db"))
TOKEN_TTL_SECONDS = 60 * 60 * 12
def _load_secret() -> bytes:
    configured = os.getenv("AUTH_SECRET")
    if configured:
        return configured.encode("utf-8")
    # A persistent local key keeps sessions valid across server restarts.
    # It lives beside the account database.
    key_path = DB_PATH.parent / "auth_secret.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        key = key_path.read_bytes()
    except FileNotFoundError:
        key = secrets.token_bytes(48)
        try:
            with key_path.open("xb") as handle:
                handle.write(key)
        except FileExistsError:  # concurrent first startup
            key = key_path.read_bytes()
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
    return key


SECRET = _load_secret()


def _connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialise() -> None:
    with _connection() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT NOT NULL UNIQUE COLLATE NOCASE,
              password_hash TEXT NOT NULL,
              security_question TEXT NOT NULL,
              security_answer_hash TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS scan_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              scan_type TEXT NOT NULL,
              concerns_json TEXT NOT NULL,
              processing_time_sec REAL,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)


def _hash(value: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt, 600_000)
    return f"{base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def _verify(value: str, stored: str) -> bool:
    try:
        salt_b64, digest_b64 = stored.split("$", 1)
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
    except (ValueError, UnicodeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt, 600_000)
    return hmac.compare_digest(actual, expected)


def create_user(email: str, password: str, security_question: str, security_answer: str) -> dict[str, Any]:
    email = email.strip().lower()
    if not email or "@" not in email or len(email) > 254:
        raise ValueError("Enter a valid email address.")
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")
    if len(security_question.strip()) < 8 or len(security_answer.strip()) < 2:
        raise ValueError("Choose a security question and enter an answer.")
    try:
        with _connection() as con:
            cur = con.execute(
                "INSERT INTO users (email, password_hash, security_question, security_answer_hash) VALUES (?, ?, ?, ?)",
                (email, _hash(password), security_question.strip(), _hash(security_answer.strip().casefold())),
            )
            return {"id": cur.lastrowid, "email": email}
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with that email already exists.") from exc


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    with _connection() as con:
        user = con.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if not user or not _verify(password, user["password_hash"]):
        return None
    return {"id": user["id"], "email": user["email"]}


def security_question(email: str) -> str | None:
    with _connection() as con:
        row = con.execute("SELECT security_question FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return row["security_question"] if row else None


def reset_password(email: str, security_answer: str, new_password: str) -> bool:
    if len(new_password) < 10:
        raise ValueError("Use a password with at least 10 characters.")
    with _connection() as con:
        row = con.execute("SELECT id, security_answer_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if not row or not _verify(security_answer.strip().casefold(), row["security_answer_hash"]):
            return False
        con.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash(new_password), row["id"]))
    return True


def issue_token(user: dict[str, Any]) -> str:
    payload = json.dumps({"sub": user["id"], "email": user["email"], "exp": int(time.time()) + TOKEN_TTL_SECONDS}, separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = hmac.new(SECRET, body, hashlib.sha256).digest()
    return f"{body.decode()}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def user_from_token(token: str) -> dict[str, Any] | None:
    try:
        body_b64, signature_b64 = token.split(".", 1)
        body = body_b64.encode()
        supplied = base64.urlsafe_b64decode(signature_b64 + "=" * (-len(signature_b64) % 4))
        expected = hmac.new(SECRET, body, hashlib.sha256).digest()
        payload = json.loads(base64.urlsafe_b64decode(body_b64 + "=" * (-len(body_b64) % 4)))
        if not hmac.compare_digest(supplied, expected) or payload["exp"] < time.time():
            return None
    except (KeyError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    with _connection() as con:
        user = con.execute("SELECT id, email FROM users WHERE id = ?", (payload["sub"],)).fetchone()
    return dict(user) if user else None


def save_history(user_id: int, result: dict[str, Any]) -> None:
    """Persist only compact measurements, never source or annotated photos."""
    with _connection() as con:
        con.execute(
            "INSERT INTO scan_history (user_id, scan_type, concerns_json, processing_time_sec) VALUES (?, ?, ?, ?)",
            (user_id, result.get("scan_type", "single_photo"), json.dumps(result["concerns"]), result.get("processing_time_sec")),
        )


def get_history(user_id: int) -> list[dict[str, Any]]:
    with _connection() as con:
        rows = con.execute(
            "SELECT id, created_at, scan_type, concerns_json, processing_time_sec FROM scan_history WHERE user_id = ? ORDER BY id DESC LIMIT 30",
            (user_id,),
        ).fetchall()
    return [{**dict(row), "concerns": json.loads(row["concerns_json"])} for row in rows]


def delete_history_entry(user_id: int, record_id: int) -> bool:
    with _connection() as con:
        cur = con.execute("DELETE FROM scan_history WHERE id = ? AND user_id = ?", (record_id, user_id))
    return cur.rowcount > 0
