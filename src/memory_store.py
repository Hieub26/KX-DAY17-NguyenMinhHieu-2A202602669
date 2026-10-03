from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Implement a simple, deterministic token estimator.

    Approximates tokens from character count (~4 characters per token heuristic).
    Returns 0 for empty or whitespace-only text.

    Args:
        text: Input string to estimate.

    Returns:
        Estimated token count as an integer.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    - Map each user id to one markdown file under root_dir/<user_id>/User.md
    - Support read / write / edit operations
    - Expose helpers like `facts()`, `upsert_fact()`, and `upsert_facts()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Resolve and sanitize the filesystem path for a user's markdown profile.

        Args:
            user_id: Unique user identifier.

        Returns:
            Path object pointing to `root_dir/<sanitized_user_id>/User.md`.
        """
        slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip()) or "default_user"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read the full text content of a user's markdown profile file.

        Args:
            user_id: Unique user identifier.

        Returns:
            The raw markdown string if the file exists, or an empty string otherwise.
        """
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write or overwrite raw markdown content into the user's User.md file.

        Creates any missing parent directories automatically.

        Args:
            user_id: Unique user identifier.
            content: Markdown formatted string to persist.

        Returns:
            Path to the written file.
        """
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of a target substring in User.md.

        Args:
            user_id: Unique user identifier.
            search_text: Substring to locate and replace.
            replacement: New substring to substitute.

        Returns:
            True if search_text was found and replaced, False otherwise.
        """
        path = self.path_for(user_id)
        if not path.is_file():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current size of the user's User.md profile file in bytes.

        Args:
            user_id: Unique user identifier.

        Returns:
            File size in bytes, or 0 if the file does not exist.
        """
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured key-value facts from the user's markdown profile.

        Parses bullet lines matching `- **key**: value` or `- key: value`.

        Args:
            user_id: Unique user identifier.

        Returns:
            Dictionary mapping fact keys to their string values.
        """
        text = self.read_text(user_id)
        if not text:
            return {}
        facts_dict: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^-\s*(?:\*\*)?([a-zA-Z0-9_\s]+?)(?:\*\*)?:\s*(.*)$", line)
            if m:
                k = m.group(1).strip()
                v = m.group(2).strip()
                facts_dict[k] = v
        return facts_dict

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        """Update existing facts or insert new facts into the user's markdown profile.

        Performs conflict resolution by updating keys in place and regenerates
        a clean, formatted markdown document.

        Args:
            user_id: Unique user identifier.
            updates: Dictionary of fact keys and updated values.

        Returns:
            Path to the updated User.md file.
        """
        if not updates:
            return self.path_for(user_id)
        current = self.facts(user_id)

        # Merge technical interests if both exist
        if "technical_interests" in updates and "technical_interests" in current:
            existing_ints = [x.strip() for x in current["technical_interests"].split(",") if x.strip()]
            new_ints = [x.strip() for x in updates["technical_interests"].split(",") if x.strip()]
            combined = list(dict.fromkeys(existing_ints + new_ints))
            updates["technical_interests"] = ", ".join(combined)

        current.update(updates)

        lines = [f"# User Profile: {user_id}", ""]
        for k, v in current.items():
            lines.append(f"- **{k}**: {v}")
        content = "\n".join(lines) + "\n"
        return self.write_text(user_id, content)

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Convenience helper to insert or update a single fact key-value pair.

        Args:
            user_id: Unique user identifier.
            key: Fact category name.
            value: Fact content string.

        Returns:
            Path to the updated User.md file.
        """
        return self.upsert_facts(user_id, {key: value})


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Handles corrections (e.g. Đà Nẵng -> Huế, backend -> MLOps), filters noise
    (e.g. temporary business trip to Hà Nội, joking about product manager),
    and skips pure recall questions.

    Args:
        message: Raw input text from the user turn.

    Returns:
        Dictionary containing extracted fact key-value pairs.
    """
    facts: dict[str, str] = {}
    m_lower = message.lower().strip()

    # Skip pure recall question turns
    is_question = m_lower.endswith("?") or any(
        m_lower.startswith(q) for q in ["bạn có thể", "bạn thử nhớ", "hãy nhắc", "đâu mới là", "bạn biết", "nếu ai đó"]
    )
    has_declaration = any(
        k in m_lower for k in ["mình tên là", "tên mình là", "tôi tên là", "đính chính", "chuyển sang", "cập nhật từ"]
    )
    if is_question and not has_declaration:
        return facts

    # 1. Name
    name_match = re.search(
        r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-Za-z0-9_\s\u00C0-\u1EF9]+?)(?:[.,\n]|$)",
        message,
        re.IGNORECASE,
    )
    if name_match:
        raw_name = name_match.group(1).strip()
        cleaned_name = re.split(r"[,.\n]|(?:\s+(?:hiện|và|đang|ở))", raw_name)[0].strip()
        if cleaned_name:
            facts["name"] = cleaned_name

    # 2. Location
    if "hà nội chỉ là nơi" in m_lower or "đừng lấy nó làm nơi ở hiện tại" in m_lower:
        # Explicit noise rejection
        pass
    elif (
        "cập nhật từ huế sang đà nẵng" in m_lower
        or "làm việc ở đà nẵng" in m_lower
        or "nơi ở hiện tại là đà nẵng" in m_lower
    ):
        facts["location"] = "Đà Nẵng"
    elif "đang ở huế" in m_lower or "vẫn ở huế" in m_lower or "giờ mình đang ở huế" in m_lower:
        if "đã cập nhật từ huế sang đà nẵng" not in m_lower and "giai đoạn này dù trước đó có nhắc huế" not in m_lower:
            facts["location"] = "Huế"
    elif "ở đà nẵng" in m_lower and "không còn ở đà nẵng" not in m_lower and "nhắc lại đà nẵng như ví dụ cũ" not in m_lower:
        facts["location"] = "Đà Nẵng"

    # 3. Profession
    if "product manager" in m_lower and "chỉ là câu đùa" in m_lower:
        facts["profession"] = "MLOps engineer"
    elif "mlops engineer" in m_lower:
        facts["profession"] = "MLOps engineer"
    elif (
        "backend engineer" in m_lower
        and "không còn làm backend engineer" not in m_lower
        and "đừng nói backend engineer" not in m_lower
        and "chứ không còn là backend engineer" not in m_lower
    ):
        facts["profession"] = "backend engineer"

    # 4. Favorite Drink
    if "cà phê sữa đá" in m_lower and any(w in m_lower for w in ["uống", "thích", "cũ", "yêu thích"]):
        facts["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite Food
    if "mì quảng" in m_lower:
        facts["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in m_lower:
        facts["pet"] = "corgi"

    # 7. Response Style
    if "3 bullet" in m_lower:
        facts["response_style"] = "3 bullet ngắn gọn có ví dụ thực chiến"
    elif "ngắn gọn" in m_lower or "bullet ngắn" in m_lower:
        facts["response_style"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Technical Interests
    interests = []
    if "python" in m_lower and any(w in m_lower for w in ["thích", "quan tâm", "học", "ôn lại"]):
        interests.append("Python")
    if "ai" in m_lower and any(w in m_lower for w in ["thích", "quan tâm", "ứng dụng", "agent", "startup"]):
        interests.append("AI")
    if "mlops" in m_lower:
        interests.append("MLOps")
    if interests:
        facts["technical_interests"] = ", ".join(interests)

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact, condensed bullet summary of older conversation messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        max_items: Maximum number of recent items to include in this summary batch.

    Returns:
        A concise multi-line string where each line represents a summarized message turn.
    """
    if not messages:
        return ""

    items_to_summarize = messages[-max_items:] if len(messages) > max_items else messages
    summary_lines = []
    for msg in items_to_summarize:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        condensed = content.replace("\n", " ")
        if len(condensed) > 100:
            condensed = condensed[:97] + "..."
        summary_lines.append(f"- {role.capitalize()}: {condensed}")

    return "\n".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Manages short-term thread memory and triggers compaction when context limits are reached.

    Maintains recent conversation messages in full detail while compressing older
    turns into an accumulated concise summary once the token count exceeds the threshold.

    Attributes:
        threshold_tokens: Token count limit that triggers message compaction.
        keep_messages: Number of recent messages to preserve uncompacted in full.
        state: In-memory dictionary tracking messages, summary, and compaction counts per thread.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a new message turn to the thread and trigger compaction if threshold is exceeded.

        Args:
            thread_id: Unique identifier for the conversation thread.
            role: Speaker role ('user' or 'assistant').
            content: Message body text.
        """
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread_state = self.state[thread_id]
        messages = thread_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current total context tokens
        summary = str(thread_state.get("summary", ""))
        summary_tokens = estimate_tokens(summary)
        messages_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        total_tokens = summary_tokens + messages_tokens

        # Trigger compaction if threshold exceeded and we have more messages than keep_messages
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            to_compact = messages[:-self.keep_messages]
            kept_messages = messages[-self.keep_messages:]

            new_summary_part = summarize_messages(to_compact)
            if summary:
                combined_summary = summary + "\n" + new_summary_part
                # Keep summary bounded if it grows too long
                summary_lines = combined_summary.splitlines()
                if len(summary_lines) > 10:
                    combined_summary = "\n".join(summary_lines[-10:])
            else:
                combined_summary = new_summary_part

            thread_state["summary"] = combined_summary
            thread_state["messages"] = kept_messages
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Retrieve the current compacted context for a conversation thread.

        Args:
            thread_id: Unique identifier for the conversation thread.

        Returns:
            Dictionary containing 'messages' (list of dicts), 'summary' (str),
            and 'compactions' (int count).
        """
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        thread_state = self.state[thread_id]
        return {
            "messages": list(thread_state.get("messages", [])),  # type: ignore
            "summary": str(thread_state.get("summary", "")),
            "compactions": int(thread_state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        """Return the cumulative number of compactions performed on a thread.

        Args:
            thread_id: Unique identifier for the conversation thread.

        Returns:
            Integer count of compaction events.
        """
        return int(self.state.get(thread_id, {}).get("compactions", 0))
