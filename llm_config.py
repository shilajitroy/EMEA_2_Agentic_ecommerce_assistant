"""Optional, centrally configured chat model for OpenRouter, Google Gemini, or Groq."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,18}\d(?!\d)")


class LLMUnavailableError(RuntimeError):
    """The optional chat model is not configured or cannot be created."""


def get_chat_model():
    """Return a LangChain chat model, or explain how to enable the optional LLM."""
    load_dotenv(PROJECT_DIR / ".env")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        raise LLMUnavailableError("No LLM provider is configured; deterministic mode is active.")

    model_name = os.getenv("LLM_MODEL", "").strip()
    if not model_name:
        raise LLMUnavailableError("Set LLM_MODEL in .env to enable the optional LLM.")

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise LLMUnavailableError("OPENROUTER_API_KEY is not set; deterministic mode is active.")
        from langchain_openai import ChatOpenAI

        extra_headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0,
            default_headers=extra_headers or None,
        )

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise LLMUnavailableError("GOOGLE_API_KEY is not set; deterministic mode is active.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not set; deterministic mode is active.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, groq_api_key=api_key, temperature=0)

    raise LLMUnavailableError(
        f"Unsupported LLM_PROVIDER '{provider}'. Choose 'openrouter', 'google', or 'groq'."
    )


def redact_sensitive_text(text: str) -> str:
    """Redact obvious email addresses and long account/card/phone-like numbers."""
    sanitized = EMAIL_PATTERN.sub("[redacted email]", text)

    def redact_number(match: re.Match[str]) -> str:
        nearby_text = sanitized[max(0, match.start() - 8):match.start()].casefold()
        if "₹" in nearby_text or "inr" in nearby_text or "rs" in nearby_text:
            return match.group(0)
        return "[redacted number]"

    return LONG_NUMBER_PATTERN.sub(redact_number, sanitized)
