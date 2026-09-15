"""SQLite 对话历史持久化测试：全部使用临时数据库，不触碰真实 data/chat_history.db"""
import app.db as db


def _set_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_chat.db"))
    db.init_db()


def test_save_and_get_history(monkeypatch, tmp_path):
    _set_tmp_db(monkeypatch, tmp_path)
    db.save_message("s1", "user", "你好")
    db.save_message("s1", "assistant", "你好，有什么可以帮你？")
    assert db.get_history("s1") == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]


def test_history_limit_returns_latest_in_order(monkeypatch, tmp_path):
    _set_tmp_db(monkeypatch, tmp_path)
    for i in range(5):
        db.save_message("s1", "user", f"q{i}")
        db.save_message("s1", "assistant", f"a{i}")
    # 最近 3 条 = a3, q4, a4，按时间正序返回
    assert db.get_history("s1", limit=3) == [
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]


def test_sessions_isolated_by_session_id(monkeypatch, tmp_path):
    _set_tmp_db(monkeypatch, tmp_path)
    db.save_message("s1", "user", "会话一")
    db.save_message("s1", "assistant", "回复一")
    db.save_message("s2", "user", "会话二")
    assert db.get_history("s1") == [
        {"role": "user", "content": "会话一"},
        {"role": "assistant", "content": "回复一"},
    ]
    assert db.get_history("s2") == [{"role": "user", "content": "会话二"}]


def test_list_sessions_by_recent_activity(monkeypatch, tmp_path):
    _set_tmp_db(monkeypatch, tmp_path)
    db.save_message("s2", "user", "第二个会话")
    db.save_message("s1", "user", "第一个会话")
    db.save_message("s1", "assistant", "回复")
    sessions = db.list_sessions()
    assert sessions[0]["session_id"] == "s1"
    assert sessions[1]["session_id"] == "s2"
    assert sessions[0]["title"] == "第一个会话"
    assert sessions[0]["msg_count"] == 2
