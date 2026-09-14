"""
SQLite 对话历史持久化
把对话历史从内存字典迁移到 SQLite 数据库，重启服务不丢数据
"""
import sqlite3
import os
import pathlib

DB_PATH = str(pathlib.Path(__file__).parent.parent / "data" / "chat_history.db")


def init_db():
    """初始化数据库，建表（如果不存在）"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
    conn.commit()
    conn.close()


def get_history(session_id, limit=20):
    """
    获取某个 session 的对话历史：取最近 limit 条消息，按时间正序返回。
    用自增 id 排序（而非 created_at，避免同秒并列导致顺序不稳）；
    先倒序取最新 N 条，再正序还原，保证多轮对话上下文始终是最新内容。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM (
            SELECT id, role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        ) ORDER BY id ASC
    """, (session_id, limit))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


def save_message(session_id, role, content):
    """保存一条消息到数据库"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()


def list_sessions():
    """
    列出所有对话会话，按最近活跃时间倒序。
    返回 [{session_id, title, msg_count}]，title 取该会话第一条用户消息。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session_id,
               (SELECT content FROM messages m2
                WHERE m2.session_id = m1.session_id AND m2.role = 'user'
                ORDER BY id ASC LIMIT 1) AS title,
               COUNT(*) AS msg_count,
               MAX(id) AS last_id
        FROM messages m1
        GROUP BY session_id
        ORDER BY last_id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {"session_id": r[0], "title": (r[1] or "新对话")[:20], "msg_count": r[2]}
        for r in rows
    ]
