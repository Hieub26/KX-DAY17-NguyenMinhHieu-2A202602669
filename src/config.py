from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from model_provider import ProviderConfig
except ImportError:
    from src.model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration parameters for the memory lab benchmarks and agents.

    Attributes:
        base_dir: Root directory of the repository.
        data_dir: Directory containing evaluation datasets (conversations.json, etc.).
        state_dir: Directory where persistent state and user profiles are stored.
        compact_threshold_tokens: Maximum token count in a thread before triggering compaction.
        compact_keep_messages: Number of recent messages to preserve uncompacted in full.
        model: ProviderConfig for the main conversational agent.
        judge_model: ProviderConfig for evaluation or judge calls if needed.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration from environment variables and return a populated LabConfig instance.

    Reads optional settings from `.env`, creates `state/` and `state/profiles/` directories
    if they do not already exist, and automatically configures provider settings.

    Args:
        base_dir: Optional root directory path. Defaults to repo root parent of `src/`.

    Returns:
        LabConfig: Fully populated configuration object for agent execution.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load from .env if present
    dotenv_path = root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    # Ensure directories exist
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Compact memory settings
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Determine main model configuration
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        if os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.getenv("CUSTOM_BASE_URL"):
            provider = "custom"
        elif os.getenv("OLLAMA_BASE_URL"):
            provider = "ollama"
        else:
            provider = "openai"

    model_defaults = {
        "openai": "gpt-4o-mini",
        "custom": "custom-model",
        "gemini": "gemini-3.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-4o-mini",
    }

    model_name = os.getenv("LLM_MODEL", model_defaults.get(provider, "gpt-4o-mini"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("CUSTOM_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
    )

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Determine judge model configuration
    judge_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_model_name = os.getenv("JUDGE_MODEL", model_defaults.get(judge_provider, model_name))
    judge_temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    judge_api_key = (
        os.getenv("JUDGE_API_KEY")
        or api_key
    )
    judge_base_url = (
        os.getenv("JUDGE_BASE_URL")
        or base_url
    )

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temperature,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config,
    )
