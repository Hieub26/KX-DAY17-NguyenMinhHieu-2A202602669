from __future__ import annotations

from dataclasses import dataclass


from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter


@dataclass
class ProviderConfig:
    """Configuration specification for model providers supported in the lab.

    Attributes:
        provider: Name of the LLM provider (e.g. 'openai', 'gemini', 'anthropic', etc.).
        model_name: Specific model identifier (e.g. 'gpt-4o-mini', 'gemini-1.5-flash').
        temperature: Sampling temperature for text generation (default 0.0 for deterministic output).
        api_key: Optional API key for authenticating with cloud provider APIs.
        base_url: Optional custom base endpoint URL (e.g. for local Ollama or OpenAI-compatible servers).
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize user-supplied provider names and aliases into canonical identifiers.

    Args:
        value: Raw provider string (e.g., 'anthorpic', 'google-genai', 'OpenAI').

    Returns:
        Canonical provider string ('openai', 'custom', 'gemini', 'anthropic', 'ollama', 'openrouter').
    """
    raw = value.strip().lower().replace("_", "-").replace(" ", "-")
    alias_map = {
        # openai
        "openai": "openai",
        "open-ai": "openai",
        "gpt": "openai",
        # custom
        "custom": "custom",
        "openai-compatible": "custom",
        "compatible": "custom",
        "local-openai": "custom",
        # gemini
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "google-generative-ai": "gemini",
        # anthropic
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        # ollama
        "ollama": "ollama",
        "local": "ollama",
        # openrouter
        "openrouter": "openrouter",
        "open-router": "openrouter",
    }
    return alias_map.get(raw, raw)


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate and return the appropriate LangChain chat model based on provider configuration.

    Supports:
        - 'openai' -> ChatOpenAI
        - 'custom' -> ChatOpenAI (with custom base_url)
        - 'gemini' -> ChatGoogleGenerativeAI
        - 'anthropic' -> ChatAnthropic
        - 'ollama' -> ChatOllama
        - 'openrouter' -> ChatOpenRouter

    Args:
        config: ProviderConfig instance containing provider name, model name, keys, and endpoints.

    Returns:
        An initialized BaseChatModel instance suitable for agent invocation.

    Raises:
        ValueError: If the normalized provider is not supported.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenRouter(**kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}' (normalized: '{provider}'). "
            f"Supported providers: openai, custom, gemini, anthropic, ollama, openrouter."
        )
