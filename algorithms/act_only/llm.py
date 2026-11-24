"""Load the project API configuration and request one action response."""

import os
import ssl

import certifi

# Use the same CA bundle as MLAQ on Windows Python 3.8.
ssl.create_default_context = (
    lambda purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None:
    ssl._create_default_https_context(purpose, cafile=certifi.where())
)
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import openai


MODEL_CONFIG = {
    "deepseek-v4-flash": ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    "deepseek-v4-pro": ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    "MiniMax-M3": ("MINIMAX_API_KEY", "MINIMAX_API_BASE", "https://api.minimax.cn/v1"),
    "gpt-4": ("OPENAI_API_KEY", "OPENAI_API_BASE", "https://api.openai.com/v1"),
    "gpt-3.5-turbo": ("OPENAI_API_KEY", "OPENAI_API_BASE", "https://api.openai.com/v1"),
    "gpt-4-1106-preview": ("OPENAI_API_KEY", "OPENAI_API_BASE", "https://api.openai.com/v1"),
    "gpt-4-0125-preview": ("OPENAI_API_KEY", "OPENAI_API_BASE", "https://api.openai.com/v1"),
    "claude": ("OPENAI_API_KEY", "OPENAI_API_BASE", "https://api.openai.com/v1"),
}


def configure_api(model, env_file):
    if model not in MODEL_CONFIG:
        raise ValueError(f"Unknown LLM model {model!r}")
    api_key_name, api_base_name, default_base = MODEL_CONFIG[model]
    settings = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip().strip('"').strip("'")
    if not settings.get(api_key_name):
        raise ValueError(f"Missing {api_key_name} in {env_file}")
    openai.api_key = settings[api_key_name]
    openai.api_base = settings.get(api_base_name, default_base)
    openai.proxy = {"http": "", "https": ""}


def query_action(prompt, model, temperature):
    chunks = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, stream=True, stream_options={"include_usage": True},
        request_timeout=45,
    )
    content = []
    usage = None
    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            if choice["delta"].get("content"):
                content.append(choice["delta"]["content"])
    return "".join(content), usage
