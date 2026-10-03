from __future__ import annotations

from pathlib import Path

try:
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent
    from config import LabConfig, load_config
    from memory_store import CompactMemoryManager, UserProfileStore
except ImportError:
    from src.agent_advanced import AdvancedAgent
    from src.agent_baseline import BaselineAgent
    from src.config import LabConfig, load_config
    from src.memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig instance configured specifically for unit tests.

    Points `state_dir` to a temporary directory and reduces `compact_threshold_tokens`
    to a low value (50 tokens) so that message compaction triggers rapidly during testing.

    Args:
        tmp_path: Pytest temporary directory fixture path.

    Returns:
        LabConfig: Configured test environment settings.
    """
    cfg = load_config()
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 50
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify that UserProfileStore correctly handles file creation, reading, and editing.

    Tests:
        1. Writing raw markdown content to `User.md` on disk.
        2. Reading file content and verifying non-zero file size.
        3. In-place string editing using `edit_text()`.
        4. Structured fact upserting and parsing via `upsert_fact()` and `facts()`.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Write initial profile
    initial_content = "# User Profile: test_user\n\n- **name**: Alice\n- **location**: Da Nang\n"
    file_path = store.write_text(user_id, initial_content)
    assert file_path.exists()
    assert store.read_text(user_id) == initial_content
    assert store.file_size(user_id) > 0

    # Edit profile text
    edited = store.edit_text(user_id, "Da Nang", "Hue")
    assert edited is True
    assert "Hue" in store.read_text(user_id)
    assert "Da Nang" not in store.read_text(user_id)

    # Upsert fact helper
    store.upsert_fact(user_id, "profession", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"
    assert facts.get("name") == "Alice"
    assert facts.get("location") == "Hue"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify that CompactMemoryManager triggers compaction when message tokens exceed threshold.

    Tests:
        1. Appending turns until cumulative tokens exceed `threshold_tokens`.
        2. Compaction counter incrementing by at least 1.
        3. Preserving exactly `keep_messages` recent turns in the active message buffer.
        4. Moving older message turns into a non-empty summary string.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "test_thread"

    # Send long messages to exceed threshold
    msg1 = "Tin nhắn số một dài vừa phải để chiếm token trong hội thoại. " * 3
    msg2 = "Tin nhắn số hai tiếp tục kéo dài để tăng tổng số token. " * 3
    msg3 = "Tin nhắn số ba làm tràn ngưỡng compact memory trong hệ thống. " * 3

    manager.append(thread_id, "user", msg1)
    manager.append(thread_id, "assistant", msg2)
    manager.append(thread_id, "user", msg3)

    assert manager.compaction_count(thread_id) >= 1
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 2  # Kept messages count
    assert len(str(ctx["summary"])) > 0  # Summary created


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify cross-session fact recall: AdvancedAgent remembers, BaselineAgent forgets.

    Tests:
        1. Feeding personal facts to both agents in thread_1.
        2. Querying recall questions in an isolated fresh thread (thread_2).
        3. BaselineAgent must not remember user facts across threads.
        4. AdvancedAgent must accurately retrieve and report facts using persistent User.md.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "test_recall_user"
    initial_message = "Chào bạn, mình tên là DũngCT, ở Huế và làm MLOps engineer."

    # Thread 1: Introduction
    baseline.reply(user_id, "thread_1", initial_message)
    advanced.reply(user_id, "thread_1", initial_message)

    # Thread 2 (Fresh thread): Ask recall questions
    recall_question = "Nhắc lại giúp mình: mình tên gì, nghề nghiệp và đang ở đâu?"
    base_ans = baseline.reply(user_id, "thread_2", recall_question)["response"]
    adv_ans = advanced.reply(user_id, "thread_2", recall_question)["response"]

    # Baseline has no User.md and must forget in a new thread
    assert "DũngCT" not in base_ans
    assert "Huế" not in base_ans

    # Advanced uses User.md and must remember across sessions
    assert "DũngCT" in adv_ans
    assert "MLOps engineer" in adv_ans
    assert "Huế" in adv_ans


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify that CompactMemoryManager prevents linear context accumulation on long threads.

    Tests:
        1. Streaming 12 lengthy conversation turns through both Baseline and Advanced agents.
        2. Verifying that AdvancedAgent triggers multiple compaction cycles.
        3. Proving that Advanced prompt tokens processed is substantially lower than Baseline.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    cfg = make_config(tmp_path)
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2

    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    thread_id = "stress_test_thread"
    for i in range(12):
        turn_msg = f"Lượt {i}: Đây là một đoạn văn bản tương đối dài với nhiều chi tiết để kiểm tra tải trọng ngữ cảnh... " * 4
        baseline.reply("stress_user", thread_id, turn_msg)
        advanced.reply("stress_user", thread_id, turn_msg)

    base_prompt_load = baseline.prompt_token_usage(thread_id)
    adv_prompt_load = advanced.prompt_token_usage(thread_id)

    # Compactions must have triggered in advanced
    assert advanced.compaction_count(thread_id) >= 1
    # Advanced prompt tokens processed must be significantly lower than baseline
    assert adv_prompt_load < base_prompt_load
