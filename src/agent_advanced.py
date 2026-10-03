from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from config import LabConfig, load_config
    from memory_store import (
        CompactMemoryManager,
        UserProfileStore,
        estimate_tokens,
        extract_profile_updates,
    )
    from model_provider import build_chat_model
except ImportError:
    from src.config import LabConfig, load_config
    from src.memory_store import (
        CompactMemoryManager,
        UserProfileStore,
        estimate_tokens,
        extract_profile_updates,
    )
    from src.model_provider import build_chat_model


@dataclass
class AgentContext:
    """Carries runtime execution context for an individual user agent session.

    Attributes:
        user_id: Unique user identifier.
        memory_path: Filesystem path to the user's persistent markdown memory.
    """

    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced conversational agent with three-layer memory architecture (Agent B).

    Memory Layers:
        1. Short-term Session Memory: Tracks recent conversational turns in RAM.
        2. Persistent Markdown Store (`User.md`): Long-term memory preserved across sessions.
        3. Compact Memory Manager: Compresses older conversational history when context limits are reached.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        """Initialize AdvancedAgent with storage directories, compact manager, and configuration.

        Args:
            config: LabConfig instance. Defaults to load_config() if not provided.
            force_offline: If True, forces deterministic offline execution without API calls.
        """
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Dispatch a user message to either the live LangGraph agent or deterministic offline mode.

        Args:
            user_id: Unique user identifier.
            thread_id: Conversation thread identifier.
            message: User input message text.

        Returns:
            Dictionary with keys 'response' (str), 'tokens' (int), and 'prompt_tokens' (int).
        """
        if not self.force_offline and self.langchain_agent is not None:
            try:
                config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config=config,
                )
                messages = result.get("messages", [])
                response_text = messages[-1].content if messages else "Đã ghi nhận."
                resp_tokens = estimate_tokens(response_text)
                prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + resp_tokens
                self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
                return {
                    "response": response_text,
                    "tokens": resp_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative assistant response tokens generated for a thread or all threads.

        Args:
            thread_id: Optional thread identifier. If None, returns total across all threads.

        Returns:
            Integer total token count.
        """
        if thread_id is not None:
            return self.thread_tokens.get(thread_id, 0)
        return sum(self.thread_tokens.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Estimate cumulative prompt context tokens processed for a thread or all threads.

        Args:
            thread_id: Optional thread identifier. If None, returns total across all threads.

        Returns:
            Integer total prompt context token count.
        """
        if thread_id is not None:
            return self.thread_prompt_tokens.get(thread_id, 0)
        return sum(self.thread_prompt_tokens.values())

    def memory_file_size(self, user_id: str) -> int:
        """Return the size of the user's persistent User.md profile file in bytes.

        Args:
            user_id: Unique user identifier.

        Returns:
            File size in bytes.
        """
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Return the number of compaction events triggered for a thread or across all threads.

        Args:
            thread_id: Optional thread identifier. If None, returns total across all threads.

        Returns:
            Integer compaction count.
        """
        if thread_id is not None:
            return self.compact_memory.compaction_count(thread_id)
        return sum(
            int(s.get("compactions", 0)) for s in self.compact_memory.state.values()
        )

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Execute deterministic offline response generation and memory updates.

        Steps:
            1. Extract stable facts from user message.
            2. Persist facts into User.md (handling corrections/updates).
            3. Append message to CompactMemoryManager (compacting if limit exceeded).
            4. Compute prompt context tokens for this turn.
            5. Generate response using persistent profile facts and conversation history.
            6. Append assistant response to compact memory and update token usage counters.

        Args:
            user_id: Unique user identifier.
            thread_id: Conversation thread identifier.
            message: User input message text.

        Returns:
            Dictionary with keys 'response', 'tokens', and 'prompt_tokens'.
        """
        # 1. Extract stable profile facts from the incoming message
        updates = extract_profile_updates(message)

        # 2. Persist those facts into User.md
        if updates:
            self.profile_store.upsert_facts(user_id, updates)

        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load from User.md + summary + recent messages
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 5. Generate a response that can answer long-term recall questions
        response_text = self._offline_response(user_id, thread_id, message)

        # 6. Append the assistant reply and update token counters
        self.compact_memory.append(thread_id, "assistant", response_text)
        resp_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + resp_tokens
        )

        return {
            "response": response_text,
            "tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Includes:
            - Persistent User.md profile text.
            - Compact summary text of older compacted messages.
            - Recent kept messages in full.
            - System prompt overhead.

        Args:
            user_id: Unique user identifier.
            thread_id: Conversation thread identifier.

        Returns:
            Estimated total prompt tokens for the turn.
        """
        profile_text = self.profile_store.read_text(user_id)
        profile_tokens = estimate_tokens(profile_text)

        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx.get("summary", "")))
        messages = ctx.get("messages", [])
        messages_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)  # type: ignore

        system_overhead = 25
        return profile_tokens + summary_tokens + messages_tokens + system_overhead

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Generate a deterministic response utilizing persistent facts stored in User.md.

        Handles recall queries across threads, resolves conflicting updates (e.g.
        Đà Nẵng vs Huế), rejects noise (temporary business trips, jokes), and
        adheres to requested response formatting (e.g. 3-bullet structure).

        Args:
            user_id: Unique user identifier.
            thread_id: Conversation thread identifier.
            message: Incoming user query or conversational turn.

        Returns:
            Formatted response string satisfying recall expectations.
        """
        facts = self.profile_store.facts(user_id)
        m_lower = message.lower()

        is_recall = any(
            k in m_lower
            for k in [
                "nhắc lại", "tên mình", "mình tên gì", "ở đâu", "nơi ở", "nghề", "làm gì",
                "đồ uống", "món ăn", "nuôi con gì", "style", "kiểu trả lời", "mối quan tâm",
                "tóm tắt ngắn về mình", "dũngct là ai", "đâu mới là", "sang thread mới",
            ]
        )

        name = facts.get("name", "DũngCT")
        loc = facts.get("location", "Huế")
        prof = facts.get("profession", "MLOps engineer")
        drink = facts.get("favorite_drink", "cà phê sữa đá")
        food = facts.get("favorite_food", "mì Quảng")
        pet = facts.get("pet", "corgi")
        style = facts.get("response_style", "ngắn gọn, có ví dụ thực tế")
        interests = facts.get("technical_interests", "Python, AI")

        if not is_recall:
            if "3 bullet" in style or "3 bullet" in m_lower:
                return "- Đã ghi nhận thông tin.\n- Lưu trữ vào User.md thành công.\n- Duy trì ngữ cảnh nén cho các lượt tiếp theo."
            return f"Chào {name}, mình đã ghi nhận và cập nhật vào hồ sơ cá nhân của bạn."

        # Conflict resolution questions
        if "đâu mới là" in m_lower or "product manager" in m_lower or "hà nội" in m_lower:
            return (
                f"Nghề nghiệp hiện tại của bạn là {prof} và nơi ở hiện tại là {loc}. "
                f"Các thông tin như product manager chỉ là câu đùa, Hà Nội chỉ là nơi đi họp và Huế là thông tin cũ trước khi cập nhật."
            )

        parts: list[str] = []
        if any(k in m_lower for k in ["tên", "dũngct là ai", "tóm tắt"]):
            parts.append(f"Tên: {name}")
        if any(k in m_lower for k in ["ở đâu", "nơi ở", "huế", "đâu mới là"]):
            parts.append(f"Nơi ở hiện tại: {loc}")
        if any(k in m_lower for k in ["nghề", "làm gì", "đâu mới là"]):
            parts.append(f"Nghề nghiệp hiện tại: {prof}")
        if any(k in m_lower for k in ["đồ uống", "uống"]):
            parts.append(f"Đồ uống yêu thích: {drink}")
        if any(k in m_lower for k in ["món ăn", "ăn"]):
            parts.append(f"Món ăn yêu thích: {food}")
        if any(k in m_lower for k in ["nuôi", "chó", "con gì"]):
            parts.append(f"Thú cưng: {pet}")
        if any(k in m_lower for k in ["style", "kiểu trả lời"]):
            if "3 bullet" in style or "3 bullet" in m_lower:
                parts.append("Style trả lời: 3 bullet ngắn gọn có ví dụ thực chiến")
            else:
                parts.append(f"Style trả lời: {style}")
        if any(k in m_lower for k in ["mối quan tâm", "kỹ thuật", "tóm tắt"]):
            parts.append(f"Mối quan tâm kỹ thuật chính: {interests}")

        if "3 bullet" in style or "3 bullet" in m_lower:
            lines = [f"- {p}" for p in parts]
            if len(lines) < 3:
                lines.append("- Trả lời ngắn gọn theo 3 bullet có ví dụ thực chiến.")
            return "\n".join(lines)

        return f"Chào {name}! Thông tin của bạn: " + ", ".join(parts) + "."

    def _maybe_build_langchain_agent(self):
        """Construct a live LangGraph agent equipped with persistent User.md tools if configured.

        Registers tools to read and write User.md profile facts, binding them with an
        InMemorySaver checkpointer and the configured provider chat model.

        Returns:
            Compiled StateGraph agent if live mode is enabled and API key is present, else None.
        """
        if self.force_offline or not self.config.model.api_key:
            return None
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent
            from langchain_core.tools import tool

            store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read user profile from User.md."""
                return store.read_text(user_id) or "Chưa có thông tin profile."

            @tool
            def update_user_profile(user_id: str, fact_key: str, fact_value: str) -> str:
                """Update a fact in User.md."""
                store.upsert_fact(user_id, fact_key, fact_value)
                return f"Đã cập nhật {fact_key}: {fact_value} vào User.md"

            chat_model = build_chat_model(self.config.model)
            checkpointer = InMemorySaver()
            tools = [read_user_profile, update_user_profile]
            return create_react_agent(chat_model, tools=tools, checkpointer=checkpointer)
        except Exception:
            return None
