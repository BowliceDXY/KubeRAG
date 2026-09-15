"""多轮对话上下文裁剪（_trim_history）测试"""
from app.rag import _trim_history


def _m(role, content):
    return {"role": role, "content": content}


def test_limits_message_count():
    history = [_m("user", f"q{i}") for i in range(10)]
    trimmed = _trim_history(history, max_messages=6, max_chars=2000)
    assert len(trimmed) <= 6


def test_limits_total_chars():
    history = [_m("user", "x" * 300) for _ in range(10)]
    trimmed = _trim_history(history, max_messages=100, max_chars=1000)
    assert sum(len(m["content"]) for m in trimmed) <= 1000


def test_keeps_at_least_recent_pair():
    history = [_m("user", "a"), _m("assistant", "b")]
    trimmed = _trim_history(history, max_messages=1, max_chars=10)
    assert len(trimmed) == 2
    assert trimmed[-1]["content"] == "b"


def test_empty_history():
    assert _trim_history([]) == []
