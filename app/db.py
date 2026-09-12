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
    """获取某个 session 的对话历史，按时间正序，默认取最近 20 条"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM messages 
        WHERE session_id = ? 
        ORDER BY created_at ASC 
        LIMIT ?
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