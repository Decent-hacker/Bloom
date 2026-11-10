import os
import subprocess
import sys
from pathlib import Path


def test_dotenv_is_loaded_before_auth_initialization(tmp_path):
    script = """
import os
import dotenv
def load_config():
    os.environ['AUTH_SECRET'] = 'test-configured-secret'
    os.environ['BLOOM_DB_PATH'] = os.environ['TEST_DB_PATH']
dotenv.load_dotenv = load_config
import main
assert main.auth_store.SECRET == b'test-configured-secret'
assert str(main.auth_store.DB_PATH) == os.environ['TEST_DB_PATH']
assert main.auth_store.DB_PATH.is_file()
"""
    env = {**os.environ, "TEST_DB_PATH": str(tmp_path / "accounts.db")}
    subprocess.run(
        [sys.executable, "-c", script], env=env,
        cwd=Path(__file__).resolve().parents[1], check=True,
        capture_output=True, text=True,
    )
