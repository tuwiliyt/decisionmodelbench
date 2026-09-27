import os

def load_env_file():
    """Load environment variables from .env file if present."""
    env_paths = [
        ".env",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.expanduser("~/.env")
    ]
    for p in env_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

load_env_file()

DEFAULT_FALLBACK_JEV_KEY = "apikey_22539801ad3ed10f4c298ead877d832613c8_cf1f0eaa3f565712acd771c6dc036e0afcfab5e5d5fc8cd56c650d48fce4b226"

JEV_API_KEY = os.environ.get("JEV_API_KEY", "").strip() or DEFAULT_FALLBACK_JEV_KEY
JEV_ENDPOINT = os.environ.get("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone").strip()
MODELS_DIR = os.environ.get("MODELS_DIR", "/root/models").strip()
