"""User accounts: sign up and log in, stored in a local SQLite file (users.db).

Passwords are NEVER stored as plain text. We store a random salt plus a
PBKDF2-SHA256 hash of the password.
"""
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).parent / "users.db"
ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the users table if it does not exist yet."""
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username      TEXT PRIMARY KEY,
                    full_name     TEXT NOT NULL,
                    email         TEXT,
                    salt          BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def create_user(username: str, full_name: str, email: str, password: str):
    """Create an account. Returns (success: bool, message: str)."""
    username = username.strip().lower()
    full_name = full_name.strip()
    email = email.strip()

    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, . _ -"
    if not full_name:
        return False, "Please enter your full name."
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "That email address doesn't look valid."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    salt = os.urandom(16)
    pw_hash = _hash_password(password, salt)
    try:
        with closing(_connect()) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO users (username, full_name, email, salt, password_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (username, full_name, email, salt, pw_hash),
                )
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created! You can now log in."


def verify_user(username: str, password: str):
    """Check a login. Returns (success: bool, user_info_or_message)."""
    username = username.strip().lower()
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT full_name, salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return False, "Incorrect username or password."
    full_name, salt, stored_hash = row
    if hmac.compare_digest(_hash_password(password, salt), stored_hash):
        return True, {"username": username, "full_name": full_name}
    return False, "Incorrect username or password."
