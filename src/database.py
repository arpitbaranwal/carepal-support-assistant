"""
SQLite Database Module for CarePal Operations (Phase 2).
Manages persistence for unanswered questions, user callback leads,
and chat conversation audits.
"""

import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class CarePalDB:
    """Thread-safe SQLite helper for CarePal operations and lead capture."""

    def __init__(self, db_path: Path = config.SQLITE_DB_PATH):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection with Row factory enabled."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Unanswered Questions Table (Operational backlog for new content)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unanswered_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    language TEXT DEFAULT 'English',
                    retrieval_score REAL DEFAULT 0.0,
                    reason TEXT DEFAULT 'out_of_scope',
                    status TEXT DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_unanswered_status ON unanswered_questions(status)")

            # 2. Leads Table (Callback requests when query is out-of-scope)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    service_needed TEXT DEFAULT 'General Inquiry',
                    query_context TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status)")

            # 3. Chat Audit Logs (Conversation tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    language TEXT DEFAULT 'English',
                    sources_cited TEXT DEFAULT '',
                    is_fallback INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # --------------------------------------------------------------------------
    # Unanswered Questions Logging
    # --------------------------------------------------------------------------
    def log_unanswered_question(
        self,
        question: str,
        language: str = "English",
        score: float = 0.0,
        reason: str = "out_of_scope",
    ) -> int:
        """Logs an unanswered question so the ops team knows what content to add."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO unanswered_questions (question, language, retrieval_score, reason, status)
                VALUES (?, ?, ?, ?, 'new')
                """,
                (question.strip(), language, float(score), reason),
            )
            conn.commit()
            return cursor.lastrowid

    def get_unanswered_questions(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Fetches logged unanswered questions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    """
                    SELECT id, question, language, retrieval_score, reason, status, created_at
                    FROM unanswered_questions
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, question, language, retrieval_score, reason, status, created_at
                    FROM unanswered_questions
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_unanswered_dataframe(self, status: Optional[str] = None, limit: int = 100):
        """Fetches logged unanswered questions as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.get_unanswered_questions(status=status, limit=limit))

    def update_unanswered_status(self, question_id: int, new_status: str) -> bool:
        """Updates status of an unanswered question (e.g. 'reviewed', 'content_added')."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE unanswered_questions SET status = ? WHERE id = ?",
                (new_status, question_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # --------------------------------------------------------------------------
    # Leads & Callbacks Capture
    # --------------------------------------------------------------------------
    def save_lead(
        self,
        name: str,
        phone: str,
        service_needed: str = "General Inquiry",
        query_context: str = "",
    ) -> int:
        """Stores a customer callback lead when a query triggers fallback."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO leads (name, phone, service_needed, query_context, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (name.strip(), phone.strip(), service_needed.strip(), query_context.strip()),
            )
            conn.commit()
            return cursor.lastrowid

    def get_leads(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Fetches callback leads."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    """
                    SELECT id, name, phone, service_needed, query_context, status, notes, created_at
                    FROM leads
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, name, phone, service_needed, query_context, status, notes, created_at
                    FROM leads
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_leads_dataframe(self, status: Optional[str] = None, limit: int = 100):
        """Fetches callback leads as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.get_leads(status=status, limit=limit))

    # Aliases for backwards compatibility and assessment naming
    get_callback_leads = get_leads
    get_callback_leads_dataframe = get_leads_dataframe
    get_unanswered_df = get_unanswered_dataframe
    get_leads_df = get_leads_dataframe

    def update_lead_status(self, lead_id: int, new_status: str, notes: Optional[str] = None) -> bool:
        """Updates status of a lead ('pending', 'contacted', 'converted', 'closed')."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if notes is not None:
                cursor.execute(
                    "UPDATE leads SET status = ?, notes = ? WHERE id = ?",
                    (new_status, notes, lead_id),
                )
            else:
                cursor.execute(
                    "UPDATE leads SET status = ? WHERE id = ?",
                    (new_status, lead_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    # --------------------------------------------------------------------------
    # Chat Audit Logging
    # --------------------------------------------------------------------------
    def log_chat(
        self,
        user_query: str,
        assistant_response: str,
        language: str = "English",
        sources_cited: str = "",
        is_fallback: bool = False,
    ) -> int:
        """Logs chat query and response for quality monitoring."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_logs (user_query, assistant_response, language, sources_cited, is_fallback)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_query.strip(), assistant_response.strip(), language, sources_cited, 1 if is_fallback else 0),
            )
            conn.commit()
            return cursor.lastrowid

    def get_chat_logs(self, limit: int = 50) -> List[Dict]:
        """Fetches recent chat logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_query, assistant_response, language, sources_cited, is_fallback, created_at
                FROM chat_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # --------------------------------------------------------------------------
    # Operations Metrics Summary
    # --------------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Computes executive metrics for the Ops Admin dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM unanswered_questions")
            total_unanswered = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM unanswered_questions WHERE status = 'new'")
            new_unanswered = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM leads")
            total_leads = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM leads WHERE status = 'pending'")
            pending_leads = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            total_chats = cursor.fetchone()[0]

            return {
                "total_unanswered": total_unanswered,
                "new_unanswered": new_unanswered,
                "total_leads": total_leads,
                "pending_leads": pending_leads,
                "total_chats": total_chats,
            }


# Singleton instance
db = CarePalDB()
