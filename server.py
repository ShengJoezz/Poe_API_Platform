from __future__ import annotations

import io
import json
import sqlite3
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = 3030
POE_BASE_URL = "https://api.poe.com/v1"
POE_ROOT_URL = "https://api.poe.com"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STREAM_DIR = DATA_DIR / "streams"
DB_PATH = DATA_DIR / "poe_audit.db"
REQUEST_TIMEOUT = 120
CHAT_HISTORY_WINDOW = 6
DEFAULT_USAGE_SYNC_LIMIT = 50
DEFAULT_USAGE_SYNC_PAGES = 2
MAX_REQUEST_BODY_BYTES = 80 * 1024 * 1024
GPT54_PRO_MODEL_ID = "gpt-5.4-pro"
FRONTEND_URL = "http://127.0.0.1:3000"
CONVERSATION_LIBRARY_LIMIT = 200
MAX_FOLDER_NAME_LENGTH = 40
GPT54_PRO_DEFAULT_PARAMS = {
    "reasoning_effort": "xhigh",
    "verbosity": "medium",
}


def utc_ms() -> int:
    return int(time.time() * 1000)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def iso_from_ms(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_ms / 1000))


def ensure_storage() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    STREAM_DIR.mkdir(exist_ok=True)


def db_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def init_db() -> None:
    ensure_storage()
    connection = db_connect()
    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                title TEXT,
                system_prompt TEXT,
                model TEXT,
                folder_id TEXT,
                latest_request_id TEXT,
                latest_response_id TEXT,
                latest_transport TEXT
            );

            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                status TEXT NOT NULL,
                reconcile_status TEXT NOT NULL,
                reconcile_note TEXT,
                transport TEXT NOT NULL,
                model TEXT NOT NULL,
                provider_response_id TEXT,
                previous_response_id TEXT,
                system_prompt TEXT,
                user_content_json TEXT NOT NULL,
                attachment_manifest_json TEXT NOT NULL,
                assistant_text TEXT NOT NULL DEFAULT '',
                assistant_reasoning_text TEXT NOT NULL DEFAULT '',
                assistant_label TEXT,
                output_file_path TEXT NOT NULL,
                input_excerpt TEXT,
                request_summary_json TEXT,
                response_meta_json TEXT,
                error_type TEXT,
                error_message TEXT,
                http_status INTEGER,
                had_done INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                started_balance INTEGER,
                ended_balance INTEGER,
                balance_delta INTEGER,
                usage_query_id TEXT,
                usage_cost_points INTEGER,
                usage_creation_time INTEGER,
                usage_type TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                rate_limit_limit INTEGER,
                rate_limit_remaining INTEGER,
                rate_limit_reset INTEGER
            );

            CREATE TABLE IF NOT EXISTS balance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                phase TEXT NOT NULL,
                captured_at INTEGER NOT NULL,
                current_point_balance INTEGER,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS usage_history_entries (
                query_id TEXT PRIMARY KEY,
                creation_time INTEGER NOT NULL,
                bot_name TEXT NOT NULL,
                usage_type TEXT,
                cost_points INTEGER,
                synced_at INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                request_id TEXT
            );

            CREATE TABLE IF NOT EXISTS request_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_requests_conversation_created_at
            ON requests(conversation_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_requests_status_reconcile
            ON requests(status, reconcile_status);

            CREATE INDEX IF NOT EXISTS idx_usage_creation_time
            ON usage_history_entries(creation_time DESC);
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        request_columns = table_columns(connection, "requests")
        if "assistant_reasoning_text" not in request_columns:
            connection.execute("ALTER TABLE requests ADD COLUMN assistant_reasoning_text TEXT NOT NULL DEFAULT ''")
        if "folder_id" not in table_columns(connection, "conversations"):
            connection.execute("ALTER TABLE conversations ADD COLUMN folder_id TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_folder_updated ON conversations(folder_id, updated_at DESC)"
        )
    connection.close()
    recover_stale_requests()


def decode_data_url_mime(data_url: str) -> str:
    if not data_url.startswith("data:"):
        return "application/octet-stream"
    prefix = data_url[5:]
    if ";" in prefix:
        return prefix.split(";", 1)[0] or "application/octet-stream"
    if "," in prefix:
        return prefix.split(",", 1)[0] or "application/octet-stream"
    return "application/octet-stream"


def data_url_payload_size(data_url: str) -> int:
    if "," not in data_url:
        return 0
    payload = data_url.split(",", 1)[1]
    length = len(payload.rstrip("="))
    return max(0, (length * 3) // 4)


def attachment_manifest(content: Any) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return manifest

    for index, part in enumerate(content):
        if not isinstance(part, dict):
            continue

        if part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url", "")).strip() if isinstance(image_url, dict) else ""
            if url:
                manifest.append(
                    {
                        "index": index,
                        "name": image_url.get("filename") or f"image-{index + 1}",
                        "kind": "image",
                        "mimeType": decode_data_url_mime(url),
                        "size": data_url_payload_size(url),
                    }
                )
            continue

        if part.get("type") == "file":
            file_part = part.get("file") or {}
            file_data = str(file_part.get("file_data", "")).strip() if isinstance(file_part, dict) else ""
            if file_data:
                mime_type = decode_data_url_mime(file_data)
                manifest.append(
                    {
                        "index": index,
                        "name": str(file_part.get("filename", f"attachment-{index + 1}")).strip() or f"attachment-{index + 1}",
                        "kind": "image" if mime_type.startswith("image/") else "file",
                        "mimeType": mime_type,
                        "size": data_url_payload_size(file_data),
                    }
                )
    return manifest


def content_excerpt(content: Any, limit: int = 160) -> str:
    if isinstance(content, str):
        return content.strip()[:limit]
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")).strip())
        return " ".join(piece for piece in texts if piece)[:limit]
    return ""


def content_is_plain_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    return all(isinstance(part, dict) and part.get("type") == "text" for part in content)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = str(part.get("text", "")).strip()
            if text:
                texts.append(text)
    return "\n\n".join(texts).strip()


def normalize_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        raise ValueError("message content must be a string or an array of content parts")

    normalized_parts: list[dict[str, Any]] = []

    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = part.get("type")
        if part_type == "text":
            text = str(part.get("text", "")).strip()
            if text:
                normalized_parts.append({"type": "text", "text": text})
            continue

        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", "")).strip()
            else:
                url = ""

            if url:
                normalized_parts.append({"type": "image_url", "image_url": {"url": url}})
            continue

        if part_type == "file":
            file_part = part.get("file")
            if isinstance(file_part, dict):
                filename = str(file_part.get("filename", "attachment")).strip() or "attachment"
                file_data = str(file_part.get("file_data", "")).strip()
            else:
                filename = "attachment"
                file_data = ""

            if file_data:
                normalized_parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": file_data,
                        },
                    }
                )
            continue

        raise ValueError(f"Unsupported content part type: {part_type}")

    return normalized_parts


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported role: {role}")

        content = normalize_content(message.get("content", ""))
        if isinstance(content, str) and content:
            normalized.append({"role": role, "content": content})
        if isinstance(content, list) and content:
            normalized.append({"role": role, "content": content})
    return normalized


def apply_model_parameter_defaults(model: str, extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    merged = dict(extra_body or {})
    if str(model).strip().lower() == GPT54_PRO_MODEL_ID:
        for key, value in GPT54_PRO_DEFAULT_PARAMS.items():
            merged.setdefault(key, value)
    return merged or None


def default_conversation_title(conversation_id: str) -> str:
    return f"Chat {conversation_id[:8]}"


def normalize_conversation_title(raw_title: str, limit: int = 40) -> str:
    title = " ".join(str(raw_title or "").split())
    if not title:
        return ""
    if len(title) <= limit:
        return title
    return f"{title[: limit - 1].rstrip()}…"


def derive_conversation_title(conversation_id: str, current_title: str | None, raw_excerpt: str | None) -> str:
    normalized_title = str(current_title or "").strip()
    excerpt_title = normalize_conversation_title(raw_excerpt or "")
    if normalized_title and normalized_title != default_conversation_title(conversation_id):
        return normalized_title
    if excerpt_title:
        return excerpt_title
    if normalized_title:
        return normalized_title
    return "新对话"


def normalize_folder_name(raw_name: Any) -> str:
    name = " ".join(str(raw_name or "").split())
    if not name:
        raise ValueError("文件夹名称不能为空")
    if len(name) > MAX_FOLDER_NAME_LENGTH:
        raise ValueError(f"文件夹名称不能超过 {MAX_FOLDER_NAME_LENGTH} 个字符")
    return name


def ensure_conversation(conversation_id: str, model: str, system_prompt: str | None) -> None:
    now = utc_ms()
    connection = db_connect()
    with connection:
        row = connection.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?, model = ?, system_prompt = ?
                WHERE id = ?
                """,
                (now, model, system_prompt or "", conversation_id),
            )
        else:
            connection.execute(
                """
                INSERT INTO conversations (id, created_at, updated_at, title, system_prompt, model, folder_id, latest_request_id, latest_response_id, latest_transport)
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
                """,
                (conversation_id, now, now, default_conversation_title(conversation_id), system_prompt or "", model),
            )
    connection.close()


def create_request_record(
    conversation_id: str,
    model: str,
    transport: str,
    system_prompt: str,
    user_content: Any,
    request_summary: dict[str, Any],
    previous_response_id: str | None,
) -> str:
    request_id = uuid.uuid4().hex
    created_at = utc_ms()
    output_file_path = STREAM_DIR / f"{request_id}.txt"
    output_file_path.write_text("", encoding="utf-8")

    connection = db_connect()
    with connection:
        conversation_row = connection.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        conversation_title = derive_conversation_title(
            conversation_id,
            conversation_row["title"] if conversation_row else None,
            content_excerpt(user_content, limit=48),
        )
        connection.execute(
            """
            INSERT INTO requests (
                id, conversation_id, created_at, updated_at, status, reconcile_status, reconcile_note,
                transport, model, provider_response_id, previous_response_id, system_prompt,
                user_content_json, attachment_manifest_json, assistant_text, assistant_reasoning_text, assistant_label,
                output_file_path, input_excerpt, request_summary_json, response_meta_json,
                error_type, error_message, http_status, had_done, cancel_requested
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, '', '', ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0, 0)
            """,
            (
                request_id,
                conversation_id,
                created_at,
                created_at,
                "queued",
                "pending",
                "等待 Usage API 对账",
                transport,
                model,
                previous_response_id,
                system_prompt,
                json_dumps(user_content),
                json_dumps(attachment_manifest(user_content)),
                model,
                str(output_file_path),
                content_excerpt(user_content),
                json_dumps(request_summary),
            ),
        )
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?, latest_request_id = ?, model = ?, system_prompt = ?, title = ?
            WHERE id = ?
            """,
            (created_at, request_id, model, system_prompt or "", conversation_title, conversation_id),
        )
    connection.close()
    return request_id


def add_request_event(request_id: str, event_type: str, payload: Any | None = None) -> None:
    connection = db_connect()
    with connection:
        connection.execute(
            """
            INSERT INTO request_events (request_id, created_at, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (request_id, utc_ms(), event_type, json_dumps(payload) if payload is not None else None),
        )
    connection.close()


def append_request_fragment(request_id: str, chunk: str, *, column_name: str, write_stream_file: bool) -> None:
    if not chunk:
        return

    connection = db_connect()
    row = connection.execute("SELECT output_file_path FROM requests WHERE id = ?", (request_id,)).fetchone()
    connection.close()
    if not row:
        return

    if write_stream_file:
        output_file_path = Path(row["output_file_path"])
        with output_file_path.open("a", encoding="utf-8") as handle:
            handle.write(chunk)

    connection = db_connect()
    with connection:
        connection.execute(
            """
            UPDATE requests
            SET {column_name} = COALESCE({column_name}, '') || ?, updated_at = ?, status = CASE WHEN status = 'queued' THEN 'streaming' ELSE status END
            WHERE id = ?
            """.format(column_name=column_name),
            (chunk, utc_ms(), request_id),
        )
    connection.close()


def append_request_output(request_id: str, chunk: str) -> None:
    append_request_fragment(request_id, chunk, column_name="assistant_text", write_stream_file=True)


def append_request_reasoning(request_id: str, chunk: str) -> None:
    append_request_fragment(request_id, chunk, column_name="assistant_reasoning_text", write_stream_file=False)


def request_has_saved_output(row: sqlite3.Row | None) -> bool:
    if not row:
        return False
    return bool(str(row["assistant_text"] or "").strip() or str(row["assistant_reasoning_text"] or "").strip())


def update_request(request_id: str, **fields: Any) -> None:
    if not fields:
        return

    assignments = []
    values = []
    for key, value in fields.items():
        assignments.append(f"{key} = ?")
        values.append(value)
    assignments.append("updated_at = ?")
    values.append(utc_ms())
    values.append(request_id)

    connection = db_connect()
    with connection:
        connection.execute(
            f"UPDATE requests SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
    connection.close()


def load_request(request_id: str) -> sqlite3.Row | None:
    connection = db_connect()
    row = connection.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    connection.close()
    return row


def refresh_conversation_record(connection: sqlite3.Connection, conversation_id: str) -> None:
    latest_row = connection.execute(
        """
        SELECT id, updated_at, provider_response_id, transport, model, system_prompt
        FROM requests
        WHERE conversation_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()

    if not latest_row:
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return

    connection.execute(
        """
        UPDATE conversations
        SET updated_at = ?, latest_request_id = ?, latest_response_id = ?, latest_transport = ?,
            model = ?, system_prompt = ?
        WHERE id = ?
        """,
        (
            latest_row["updated_at"],
            latest_row["id"],
            latest_row["provider_response_id"],
            latest_row["transport"],
            latest_row["model"],
            latest_row["system_prompt"] or "",
            conversation_id,
        ),
    )


def delete_request_record(request_id: str) -> sqlite3.Row | None:
    row = load_request(request_id)
    if not row:
        return None

    output_file_path = Path(row["output_file_path"])
    connection = db_connect()
    with connection:
        connection.execute("DELETE FROM balance_snapshots WHERE request_id = ?", (request_id,))
        connection.execute("DELETE FROM request_events WHERE request_id = ?", (request_id,))
        connection.execute("UPDATE usage_history_entries SET request_id = NULL WHERE request_id = ?", (request_id,))
        if row["usage_query_id"]:
            connection.execute(
                "UPDATE usage_history_entries SET request_id = NULL WHERE query_id = ?",
                (row["usage_query_id"],),
            )
        connection.execute("DELETE FROM requests WHERE id = ?", (request_id,))
        refresh_conversation_record(connection, row["conversation_id"])
    connection.close()

    if output_file_path.exists():
        output_file_path.unlink()

    return row


def latest_balance_snapshot() -> sqlite3.Row | None:
    connection = db_connect()
    row = connection.execute(
        "SELECT * FROM balance_snapshots ORDER BY captured_at DESC, id DESC LIMIT 1"
    ).fetchone()
    connection.close()
    return row


def record_balance_snapshot(
    request_id: str | None,
    phase: str,
    current_point_balance: int | None,
    raw_payload: dict[str, Any],
) -> None:
    timestamp = utc_ms()
    connection = db_connect()
    with connection:
        connection.execute(
            """
            INSERT INTO balance_snapshots (request_id, phase, captured_at, current_point_balance, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (request_id, phase, timestamp, current_point_balance, json_dumps(raw_payload)),
        )

        if request_id and phase == "before":
            connection.execute(
                "UPDATE requests SET started_balance = ?, updated_at = ? WHERE id = ?",
                (current_point_balance, timestamp, request_id),
            )
        elif request_id and phase == "after":
            row = connection.execute(
                "SELECT started_balance FROM requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            started_balance = row["started_balance"] if row else None
            balance_delta = None
            if started_balance is not None and current_point_balance is not None:
                balance_delta = started_balance - current_point_balance
            connection.execute(
                """
                UPDATE requests
                SET ended_balance = ?, balance_delta = ?, updated_at = ?
                WHERE id = ?
                """,
                (current_point_balance, balance_delta, timestamp, request_id),
            )
    connection.close()
def extract_error_payload(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        return {"message": "Poe API returned an empty error response."}

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"message": raw_text}

    error = parsed.get("error", {})
    if isinstance(error, dict) and error.get("message"):
        return {
            "message": error.get("message"),
            "type": error.get("type"),
            "code": error.get("code"),
            "metadata": error.get("metadata"),
        }
    return {"message": raw_text}


def summarize_reconcile(
    started_balance: int | None,
    ended_balance: int | None,
    cost_points: int | None,
) -> tuple[str, str]:
    if cost_points is None:
        return "pending", "等待 Usage API 条目匹配"
    if started_balance is None or ended_balance is None:
        return "matched", "已匹配 Usage API 条目，但没有完整的前后余额快照"

    observed_delta = started_balance - ended_balance
    if observed_delta == cost_points:
        return "matched", f"对账成功，余额变化与 cost_points 一致（{cost_points}）"
    return "unmatched", f"已匹配 Usage API 条目，但余额变化 {observed_delta} 与 cost_points {cost_points} 不一致"


def normalize_model_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def upsert_usage_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return

    synced_at = utc_ms()
    connection = db_connect()
    with connection:
        for entry in entries:
            connection.execute(
                """
                INSERT INTO usage_history_entries (query_id, creation_time, bot_name, usage_type, cost_points, synced_at, raw_json, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT request_id FROM usage_history_entries WHERE query_id = ?), NULL))
                ON CONFLICT(query_id) DO UPDATE SET
                    creation_time = excluded.creation_time,
                    bot_name = excluded.bot_name,
                    usage_type = excluded.usage_type,
                    cost_points = excluded.cost_points,
                    synced_at = excluded.synced_at,
                    raw_json = excluded.raw_json
                """,
                (
                    entry.get("query_id"),
                    entry.get("creation_time"),
                    entry.get("bot_name"),
                    entry.get("usage_type"),
                    entry.get("cost_points"),
                    synced_at,
                    json_dumps(entry),
                    entry.get("query_id"),
                ),
            )
    connection.close()


def reconcile_requests() -> None:
    connection = db_connect()
    with connection:
        request_rows = connection.execute(
            """
            SELECT *
            FROM requests
            WHERE status IN ('completed', 'partial_saved', 'timed_out', 'failed')
            ORDER BY created_at DESC
            """
        ).fetchall()

        used_query_ids = {row["usage_query_id"] for row in request_rows if row["usage_query_id"]}
        usage_rows = connection.execute(
            "SELECT * FROM usage_history_entries ORDER BY creation_time DESC"
        ).fetchall()
        assigned_query_ids = set(used_query_ids)

        for request in request_rows:
            request_id = request["id"]
            if request["usage_query_id"]:
                note_status, note = summarize_reconcile(
                    request["started_balance"],
                    request["ended_balance"],
                    request["usage_cost_points"],
                )
                connection.execute(
                    """
                    UPDATE requests
                    SET reconcile_status = ?, reconcile_note = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (note_status, note, utc_ms(), request_id),
                )
                continue

            request_time_us = int(request["created_at"]) * 1000
            best_match = None
            best_distance = None

            for usage in usage_rows:
                query_id = usage["query_id"]
                if query_id in assigned_query_ids:
                    continue
                if normalize_model_name(usage["bot_name"]) != normalize_model_name(request["model"]):
                    continue

                distance = abs(int(usage["creation_time"]) - request_time_us)
                if distance > 15 * 60 * 1_000_000:
                    continue

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match = usage

            if best_match:
                assigned_query_ids.add(best_match["query_id"])
                reconcile_status, reconcile_note = summarize_reconcile(
                    request["started_balance"],
                    request["ended_balance"],
                    best_match["cost_points"],
                )
                connection.execute(
                    """
                    UPDATE requests
                    SET usage_query_id = ?, usage_cost_points = ?, usage_creation_time = ?, usage_type = ?,
                        reconcile_status = ?, reconcile_note = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        best_match["query_id"],
                        best_match["cost_points"],
                        best_match["creation_time"],
                        best_match["usage_type"],
                        reconcile_status,
                        reconcile_note,
                        utc_ms(),
                        request_id,
                    ),
                )
                connection.execute(
                    "UPDATE usage_history_entries SET request_id = ? WHERE query_id = ?",
                    (request_id, best_match["query_id"]),
                )
                continue

            age_ms = utc_ms() - int(request["created_at"])
            if age_ms > 5 * 60 * 1000:
                connection.execute(
                    """
                    UPDATE requests
                    SET reconcile_status = 'unmatched', reconcile_note = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    ("5 分钟内未找到匹配的 Usage API 条目", utc_ms(), request_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE requests
                    SET reconcile_status = 'pending', reconcile_note = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    ("等待 Usage API 条目同步", utc_ms(), request_id),
                )
    connection.close()


CONVERSATION_OVERVIEW_SELECT = """
    SELECT c.id,
           c.created_at,
           c.updated_at,
           c.title,
           c.system_prompt,
           c.model,
           c.folder_id,
           c.latest_request_id,
           f.name AS folder_name,
           r.input_excerpt AS latest_excerpt,
           r.usage_cost_points AS latest_cost_points,
           r.status AS latest_status,
           (SELECT COUNT(*) FROM requests req WHERE req.conversation_id = c.id) AS request_count
    FROM conversations c
    LEFT JOIN folders f ON f.id = c.folder_id
    LEFT JOIN requests r ON r.id = c.latest_request_id
"""


def folder_payload_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "createdAt": row["created_at"],
        "createdAtLabel": iso_from_ms(row["created_at"]),
        "updatedAt": row["updated_at"],
        "updatedAtLabel": iso_from_ms(row["updated_at"]),
        "conversationCount": row["conversation_count"],
    }


def folders_payload() -> list[dict[str, Any]]:
    connection = db_connect()
    rows = connection.execute(
        """
        SELECT f.id,
               f.name,
               f.created_at,
               MAX(COALESCE(c.updated_at, f.updated_at)) AS updated_at,
               COUNT(c.id) AS conversation_count
        FROM folders f
        LEFT JOIN conversations c ON c.folder_id = f.id
        GROUP BY f.id, f.name, f.created_at, f.updated_at
        ORDER BY LOWER(f.name) ASC, f.created_at ASC
        """
    ).fetchall()
    connection.close()
    return [folder_payload_from_row(row) for row in rows]


def conversation_payload_from_row(row: sqlite3.Row) -> dict[str, Any]:
    request_count = int(row["request_count"] or 0)
    latest_excerpt = row["latest_excerpt"] or ""
    title = derive_conversation_title(row["id"], row["title"], latest_excerpt)
    return {
        "id": row["id"],
        "title": title,
        "createdAt": row["created_at"],
        "createdAtLabel": iso_from_ms(row["created_at"]),
        "updatedAt": row["updated_at"],
        "updatedAtLabel": iso_from_ms(row["updated_at"]),
        "model": row["model"],
        "systemPrompt": row["system_prompt"] or "",
        "folderId": row["folder_id"],
        "folderName": row["folder_name"],
        "latestRequestId": row["latest_request_id"],
        "latestExcerpt": latest_excerpt,
        "latestStatus": row["latest_status"] or "local",
        "latestCostPoints": row["latest_cost_points"],
        "requestCount": request_count,
        "isEmpty": request_count == 0,
    }


def conversation_library(limit: int = CONVERSATION_LIBRARY_LIMIT) -> list[dict[str, Any]]:
    connection = db_connect()
    rows = connection.execute(
        f"""
        {CONVERSATION_OVERVIEW_SELECT}
        ORDER BY c.updated_at DESC, c.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()
    return [conversation_payload_from_row(row) for row in rows]


def conversation_overview(conversation_id: str) -> dict[str, Any] | None:
    connection = db_connect()
    row = connection.execute(
        f"""
        {CONVERSATION_OVERVIEW_SELECT}
        WHERE c.id = ?
        """,
        (conversation_id,),
    ).fetchone()
    connection.close()
    return conversation_payload_from_row(row) if row else None


def create_folder_record(raw_name: Any) -> dict[str, Any]:
    folder_id = uuid.uuid4().hex
    now = utc_ms()
    name = normalize_folder_name(raw_name)
    connection = db_connect()
    try:
        with connection:
            connection.execute(
                "INSERT INTO folders (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (folder_id, name, now, now),
            )
    except sqlite3.IntegrityError as error:
        connection.close()
        raise ValueError("文件夹名称已存在") from error
    connection.close()

    for folder in folders_payload():
        if folder["id"] == folder_id:
            return folder
    raise RuntimeError("Folder creation succeeded but could not be reloaded")


def delete_folder_record(folder_id: str) -> dict[str, Any] | None:
    connection = db_connect()
    row = connection.execute(
        """
        SELECT f.id, f.name, COUNT(c.id) AS conversation_count
        FROM folders f
        LEFT JOIN conversations c ON c.folder_id = f.id
        WHERE f.id = ?
        GROUP BY f.id, f.name
        """,
        (folder_id,),
    ).fetchone()
    if not row:
        connection.close()
        return None

    with connection:
        connection.execute("UPDATE conversations SET folder_id = NULL WHERE folder_id = ?", (folder_id,))
        connection.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
    connection.close()
    return {
        "id": row["id"],
        "name": row["name"],
        "conversationCount": row["conversation_count"],
    }


def assign_conversation_folder(
    conversation_id: str,
    folder_id: str | None,
    model: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    now = utc_ms()
    connection = db_connect()
    with connection:
        if folder_id:
            folder_row = connection.execute("SELECT id FROM folders WHERE id = ?", (folder_id,)).fetchone()
            if not folder_row:
                raise LookupError("文件夹不存在")
            connection.execute("UPDATE folders SET updated_at = ? WHERE id = ?", (now, folder_id))

        conversation_row = connection.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if conversation_row:
            connection.execute(
                "UPDATE conversations SET folder_id = ? WHERE id = ?",
                (folder_id, conversation_id),
            )
        else:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, created_at, updated_at, title, system_prompt, model, folder_id,
                    latest_request_id, latest_response_id, latest_transport
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    conversation_id,
                    now,
                    now,
                    "新对话",
                    system_prompt or "",
                    model or GPT54_PRO_MODEL_ID,
                    folder_id,
                ),
            )
    connection.close()

    overview = conversation_overview(conversation_id)
    if not overview:
        raise RuntimeError("Conversation assignment succeeded but could not be reloaded")
    return overview


def conversation_rows(conversation_id: str) -> list[sqlite3.Row]:
    connection = db_connect()
    rows = connection.execute(
        """
        SELECT *
        FROM requests
        WHERE conversation_id = ?
        ORDER BY created_at ASC
        """,
        (conversation_id,),
    ).fetchall()
    connection.close()
    return rows


def request_to_message_pair(row: sqlite3.Row) -> list[dict[str, Any]]:
    user_content = json.loads(row["user_content_json"]) if row["user_content_json"] else ""
    return [
        {
            "messageId": f"{row['id']}:user",
            "requestId": row["id"],
            "role": "user",
            "label": "你",
            "content": user_content,
            "status": row["status"],
            "createdAt": row["created_at"],
            "reconcileStatus": row["reconcile_status"],
            "transport": row["transport"],
        },
        {
            "messageId": f"{row['id']}:assistant",
            "requestId": row["id"],
            "role": "assistant",
            "label": row["assistant_label"] or row["model"],
            "content": row["assistant_text"] or "",
            "reasoningContent": row["assistant_reasoning_text"] or "",
            "status": row["status"],
            "createdAt": row["updated_at"],
            "reconcileStatus": row["reconcile_status"],
            "transport": row["transport"],
        },
    ]


def conversation_transcript(conversation_id: str) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for row in conversation_rows(conversation_id):
        transcript.extend(request_to_message_pair(row))
    return transcript


def recent_requests(limit: int = 40) -> list[dict[str, Any]]:
    connection = db_connect()
    rows = connection.execute(
        """
        SELECT id, conversation_id, created_at, updated_at, status, reconcile_status, reconcile_note,
               transport, model, input_excerpt, had_done, started_balance, ended_balance, balance_delta,
               usage_query_id, usage_cost_points, usage_creation_time, http_status, error_message
        FROM requests
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()

    payload = []
    for row in rows:
        payload.append(
            {
                "id": row["id"],
                "conversationId": row["conversation_id"],
                "createdAt": row["created_at"],
                "createdAtLabel": iso_from_ms(row["created_at"]),
                "updatedAt": row["updated_at"],
                "status": row["status"],
                "reconcileStatus": row["reconcile_status"],
                "reconcileNote": row["reconcile_note"],
                "transport": row["transport"],
                "model": row["model"],
                "excerpt": row["input_excerpt"] or "",
                "hadDone": bool(row["had_done"]),
                "startedBalance": row["started_balance"],
                "endedBalance": row["ended_balance"],
                "balanceDelta": row["balance_delta"],
                "usageQueryId": row["usage_query_id"],
                "usageCostPoints": row["usage_cost_points"],
                "usageCreationTime": row["usage_creation_time"],
                "httpStatus": row["http_status"],
                "errorMessage": row["error_message"],
            }
        )
    return payload


def audit_summary() -> dict[str, Any]:
    connection = db_connect()
    status_counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM requests GROUP BY status"
        ).fetchall()
    }
    reconcile_counts = {
        row["reconcile_status"]: row["count"]
        for row in connection.execute(
            "SELECT reconcile_status, COUNT(*) AS count FROM requests GROUP BY reconcile_status"
        ).fetchall()
    }
    usage_sync = connection.execute(
        "SELECT MAX(synced_at) AS synced_at, COUNT(*) AS count FROM usage_history_entries"
    ).fetchone()
    connection.close()

    latest_balance = latest_balance_snapshot()
    return {
        "latestBalance": {
            "currentPointBalance": latest_balance["current_point_balance"],
            "capturedAt": latest_balance["captured_at"],
            "capturedAtLabel": iso_from_ms(latest_balance["captured_at"]),
        }
        if latest_balance
        else None,
        "requestStatusCounts": status_counts,
        "reconcileCounts": reconcile_counts,
        "usageSync": {
            "lastSyncedAt": usage_sync["synced_at"] if usage_sync else None,
            "lastSyncedAtLabel": iso_from_ms(usage_sync["synced_at"]) if usage_sync and usage_sync["synced_at"] else None,
            "entryCount": usage_sync["count"] if usage_sync else 0,
        },
    }


def state_payload(conversation_id: str) -> dict[str, Any]:
    connection = db_connect()
    conversation = connection.execute(
        "SELECT * FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    connection.close()
    overview = conversation_overview(conversation_id)

    return {
        "conversation": {
            "id": conversation_id,
            "exists": conversation is not None,
            "title": overview["title"] if overview else derive_conversation_title(conversation_id, conversation["title"] if conversation else None, ""),
            "latestResponseId": conversation["latest_response_id"] if conversation else None,
            "latestTransport": conversation["latest_transport"] if conversation else None,
            "model": overview["model"] if overview else (conversation["model"] if conversation else None),
            "systemPrompt": overview["systemPrompt"] if overview else (conversation["system_prompt"] if conversation else ""),
            "folderId": overview["folderId"] if overview else (conversation["folder_id"] if conversation else None),
        },
        "transcript": conversation_transcript(conversation_id),
        "folders": folders_payload(),
        "conversations": conversation_library(),
        "recentRequests": recent_requests(),
        "summary": audit_summary(),
    }


def build_evidence_zip(request_id: str) -> bytes:
    row = load_request(request_id)
    if not row:
        raise FileNotFoundError("Request not found")

    connection = db_connect()
    balance_rows = connection.execute(
        "SELECT * FROM balance_snapshots WHERE request_id = ? ORDER BY captured_at ASC, id ASC",
        (request_id,),
    ).fetchall()
    event_rows = connection.execute(
        "SELECT * FROM request_events WHERE request_id = ? ORDER BY created_at ASC, id ASC",
        (request_id,),
    ).fetchall()
    usage_row = None
    if row["usage_query_id"]:
        usage_row = connection.execute(
            "SELECT * FROM usage_history_entries WHERE query_id = ?",
            (row["usage_query_id"],),
        ).fetchone()
    connection.close()

    summary_markdown = "\n".join(
        [
            f"# Evidence Pack: {request_id}",
            "",
            f"- Conversation: `{row['conversation_id']}`",
            f"- Model: `{row['model']}`",
            f"- Transport: `{row['transport']}`",
            f"- Status: `{row['status']}`",
            f"- Reconcile: `{row['reconcile_status']}`",
            f"- Reconcile note: {row['reconcile_note'] or 'n/a'}",
            f"- Created at: {iso_from_ms(row['created_at'])}",
            f"- Updated at: {iso_from_ms(row['updated_at'])}",
            f"- Started balance: {row['started_balance']}",
            f"- Ended balance: {row['ended_balance']}",
            f"- Balance delta: {row['balance_delta']}",
            f"- Usage query id: {row['usage_query_id'] or 'n/a'}",
            f"- Usage cost points: {row['usage_cost_points']}",
            f"- HTTP status: {row['http_status']}",
            f"- Error: {row['error_message'] or 'n/a'}",
            f"- Assistant chars: {len(row['assistant_text'] or '')}",
            f"- Reasoning chars: {len(row['assistant_reasoning_text'] or '')}",
        ]
    )

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.md", summary_markdown)
        archive.writestr("request.json", compact_json({key: row[key] for key in row.keys()}))
        archive.writestr("transcript.json", compact_json(conversation_transcript(row["conversation_id"])))
        archive.writestr(
            "balance_snapshots.json",
            compact_json([{key: item[key] for key in item.keys()} for item in balance_rows]),
        )
        archive.writestr(
            "request_events.json",
            compact_json([{key: item[key] for key in item.keys()} for item in event_rows]),
        )
        archive.writestr(
            "usage_entry.json",
            compact_json({key: usage_row[key] for key in usage_row.keys()}) if usage_row else "{}",
        )

        output_file_path = Path(row["output_file_path"])
        if output_file_path.exists():
            archive.writestr("stream_output.txt", output_file_path.read_text(encoding="utf-8"))
        if row["assistant_reasoning_text"]:
            archive.writestr("reasoning_output.txt", row["assistant_reasoning_text"])

    payload.seek(0)
    return payload.read()


def header_int(headers: Any, name: str) -> int | None:
    raw_value = headers.get(name)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def rate_limit_fields_from_headers(headers: Any) -> dict[str, Any]:
    return {
        "rate_limit_limit": header_int(headers, "x-ratelimit-limit-requests"),
        "rate_limit_remaining": header_int(headers, "x-ratelimit-remaining-requests"),
        "rate_limit_reset": header_int(headers, "x-ratelimit-reset-requests"),
    }


def fetch_json(
    url: str,
    api_key: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[dict[str, Any], Any]:
    request = urllib.request.Request(
        url=url,
        data=json_dumps(body).encode("utf-8") if body is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method=method,
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    raw_payload = response.read()
    payload = json.loads(raw_payload.decode("utf-8")) if raw_payload else {}
    return payload, response


def fetch_current_balance(api_key: str) -> dict[str, Any]:
    payload, response = fetch_json(f"{POE_ROOT_URL}/usage/current_balance", api_key)
    return {
        "payload": payload,
        "balance": payload.get("current_point_balance"),
        "headers": rate_limit_fields_from_headers(response.headers),
    }


def fetch_usage_history_page(
    api_key: str,
    limit: int = DEFAULT_USAGE_SYNC_LIMIT,
    starting_after: str | None = None,
) -> dict[str, Any]:
    query = {"limit": limit}
    if starting_after:
        query["starting_after"] = starting_after
    payload, response = fetch_json(
        f"{POE_ROOT_URL}/usage/points_history?{urllib.parse.urlencode(query)}",
        api_key,
    )
    return {
        "payload": payload,
        "headers": rate_limit_fields_from_headers(response.headers),
    }


def sync_usage_history(
    api_key: str,
    pages: int = DEFAULT_USAGE_SYNC_PAGES,
    limit: int = DEFAULT_USAGE_SYNC_LIMIT,
) -> dict[str, Any]:
    all_entries: list[dict[str, Any]] = []
    starting_after = None
    has_more = False

    for _ in range(max(1, pages)):
        page = fetch_usage_history_page(api_key, limit=limit, starting_after=starting_after)
        payload = page["payload"]
        entries = payload.get("data") or []
        if isinstance(entries, list):
            all_entries.extend(item for item in entries if isinstance(item, dict))
        has_more = bool(payload.get("has_more"))
        if not has_more or not entries:
            break
        starting_after = entries[-1].get("query_id")
        if not starting_after:
            break

    upsert_usage_entries(all_entries)
    reconcile_requests()
    return {"entriesFetched": len(all_entries), "hasMore": has_more}


def build_chat_history_messages(conversation_id: str, current_user_content: Any, system_prompt: str) -> list[dict[str, Any]]:
    rows = conversation_rows(conversation_id)[-CHAT_HISTORY_WINDOW:]
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for row in rows:
        previous_user_content = json.loads(row["user_content_json"]) if row["user_content_json"] else ""
        messages.append({"role": "user", "content": previous_user_content})
        assistant_text = row["assistant_text"] or ""
        if assistant_text.strip():
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": current_user_content})
    return messages


def latest_response_id(conversation_id: str, model: str) -> str | None:
    connection = db_connect()
    row = connection.execute(
        """
        SELECT latest_response_id, model, latest_transport
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,),
    ).fetchone()
    connection.close()

    if not row:
        return None
    if row["model"] != model or row["latest_transport"] != "responses":
        return None
    return row["latest_response_id"]


def response_transport_for_content(content: Any) -> str:
    return "responses" if content_is_plain_text(content) else "chat_completions"


def extract_response_output_text(response_object: Any) -> str:
    if not isinstance(response_object, dict):
        return ""

    direct_text = response_object.get("output_text")
    if isinstance(direct_text, str) and direct_text:
        return direct_text

    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = str(node.get("type", ""))
            text_value = node.get("text")
            if node_type in {"output_text", "text"} and isinstance(text_value, str):
                parts.append(text_value)
            for key in ("content", "output", "items", "message"):
                if key in node:
                    walk(node[key])
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(response_object.get("output"))
    return "".join(parts)


def restore_request_artifacts_from_events(
    request_id: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    own_connection = connection is None
    if own_connection:
        connection = db_connect()

    assert connection is not None
    event_rows = connection.execute(
        "SELECT event_type, payload_json FROM request_events WHERE request_id = ? ORDER BY created_at ASC, id ASC",
        (request_id,),
    ).fetchall()

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    provider_response_id = None
    had_done = False

    for row in event_rows:
        event_type = row["event_type"]
        if event_type == "provider.done":
            had_done = True
            continue

        payload_json = row["payload_json"]
        if not payload_json:
            continue

        try:
            parsed = json.loads(payload_json)
        except json.JSONDecodeError:
            continue

        if event_type == "chat.chunk" and isinstance(parsed, dict):
            provider_response_id = provider_response_id or parsed.get("id")
            choice = (parsed.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content_piece = delta.get("content")
            reasoning_piece = delta.get("reasoning_content")
            if isinstance(content_piece, str):
                content_parts.append(content_piece)
            if isinstance(reasoning_piece, str):
                reasoning_parts.append(reasoning_piece)
            if choice.get("finish_reason") is not None:
                had_done = True
            continue

        if event_type == "response.output_text.delta" and isinstance(parsed, dict):
            delta = parsed.get("delta")
            if isinstance(delta, str):
                content_parts.append(delta)
            continue

        if event_type in {"response.completed", "response.failed", "response.incomplete"} and isinstance(parsed, dict):
            response_object = parsed.get("response") or {}
            provider_response_id = provider_response_id or response_object.get("id")
            if event_type == "response.completed":
                had_done = True
                if not "".join(content_parts).strip():
                    completed_text = extract_response_output_text(response_object)
                    if completed_text:
                        content_parts.append(completed_text)

    result = {
        "assistant_text": "".join(content_parts),
        "assistant_reasoning_text": "".join(reasoning_parts),
        "provider_response_id": provider_response_id,
        "had_done": had_done,
        "event_count": len(event_rows),
    }

    if own_connection:
        connection.close()
    return result


def recover_stale_requests() -> None:
    connection = db_connect()
    rows = connection.execute(
        """
        SELECT *
        FROM requests
        WHERE status IN ('queued', 'streaming')
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()

    now = utc_ms()
    with connection:
        for row in rows:
            restored = restore_request_artifacts_from_events(row["id"], connection=connection)
            assistant_text = restored["assistant_text"] or (row["assistant_text"] or "")
            reasoning_text = restored["assistant_reasoning_text"] or (row["assistant_reasoning_text"] or "")
            had_done = bool(row["had_done"]) or bool(restored["had_done"])
            provider_response_id = row["provider_response_id"] or restored["provider_response_id"]
            has_saved_output = bool(assistant_text.strip() or reasoning_text.strip())

            fields: dict[str, Any] = {}
            if assistant_text != (row["assistant_text"] or ""):
                fields["assistant_text"] = assistant_text
                Path(row["output_file_path"]).write_text(assistant_text, encoding="utf-8")
            if reasoning_text != (row["assistant_reasoning_text"] or ""):
                fields["assistant_reasoning_text"] = reasoning_text
            if provider_response_id and provider_response_id != row["provider_response_id"]:
                fields["provider_response_id"] = provider_response_id
            if had_done and not row["had_done"]:
                fields["had_done"] = 1

            age_ms = now - int(row["updated_at"] or row["created_at"])
            if row["cancel_requested"]:
                fields.setdefault("status", "partial_saved" if has_saved_output else "failed")
                fields.setdefault("error_type", row["error_type"] or "cancelled_by_user")
                fields.setdefault(
                    "error_message",
                    row["error_message"] or "本地会话在取消期间中断，已按已保存片段恢复。",
                )
            elif age_ms > 1000:
                fields.setdefault(
                    "status",
                    "completed" if had_done and assistant_text.strip() else ("partial_saved" if has_saved_output else "failed"),
                )
                if fields["status"] != "completed":
                    fields.setdefault("error_type", row["error_type"] or "local_session_interrupted")
                    fields.setdefault(
                        "error_message",
                        row["error_message"] or "本地流式会话中断，已按已保存事件恢复。",
                    )

            if not fields:
                continue

            assignments = []
            values = []
            for key, value in fields.items():
                assignments.append(f"{key} = ?")
                values.append(value)
            assignments.append("updated_at = ?")
            values.append(utc_ms())
            values.append(row["id"])
            connection.execute(
                f"UPDATE requests SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

    connection.close()


def iter_sse_data_lines(response: Any) -> Any:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


class ActiveRequest:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.cancel_event = threading.Event()


ACTIVE_REQUESTS: dict[str, ActiveRequest] = {}
ACTIVE_REQUESTS_LOCK = threading.Lock()


def register_active_request(request_id: str) -> ActiveRequest:
    active_request = ActiveRequest(request_id)
    with ACTIVE_REQUESTS_LOCK:
        ACTIVE_REQUESTS[request_id] = active_request
    return active_request


def get_active_request(request_id: str) -> ActiveRequest | None:
    with ACTIVE_REQUESTS_LOCK:
        return ACTIVE_REQUESTS.get(request_id)


def unregister_active_request(request_id: str) -> None:
    with ACTIVE_REQUESTS_LOCK:
        ACTIVE_REQUESTS.pop(request_id, None)


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(f"{self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/state":
            self.handle_state(parsed)
            return
        if parsed.path.startswith("/api/requests/") and parsed.path.endswith("/evidence.zip"):
            self.handle_evidence_download(parsed.path)
            return
        if parsed.path in {"/", "/index.html"}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", FRONTEND_URL)
            self.end_headers()
            return
        if parsed.path in {"/favicon.ico", "/favicon.svg"}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", f"{FRONTEND_URL}/favicon.svg")
            self.end_headers()
            return
        self.send_json(
            HTTPStatus.NOT_FOUND,
            {"error": "Not found", "hint": f"Open the React app at {FRONTEND_URL}."},
        )

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            self.handle_chat()
            return
        if parsed.path == "/api/folders":
            self.handle_create_folder()
            return
        if parsed.path == "/api/sync":
            self.handle_sync()
            return
        if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/folder"):
            self.handle_assign_conversation_folder(parsed.path)
            return
        if parsed.path.startswith("/api/requests/") and parsed.path.endswith("/cancel"):
            self.handle_cancel(parsed.path)
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/requests/"):
            self.handle_delete_request(parsed.path)
            return
        if parsed.path.startswith("/api/folders/"):
            self.handle_delete_folder(parsed.path)
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json_dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_sse(self, event: str, payload: dict[str, Any]) -> None:
        body = f"event: {event}\ndata: {json_dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def safe_send_sse(self, event: str, payload: dict[str, Any], client_state: dict[str, bool]) -> None:
        if not client_state["connected"]:
            return
        try:
            self.send_sse(event, payload)
        except (BrokenPipeError, ConnectionResetError):
            client_state["connected"] = False

    def read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_REQUEST_BODY_BYTES:
            raise ValueError("Request body too large")
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        if not raw_body:
            return {}
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def handle_state(self, parsed: urllib.parse.ParseResult) -> None:
        query = urllib.parse.parse_qs(parsed.query)
        conversation_id = (query.get("conversation_id") or [""])[0].strip()
        if not conversation_id:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "conversation_id is required"})
            return
        self.send_json(HTTPStatus.OK, state_payload(conversation_id))

    def handle_sync(self) -> None:
        try:
            payload = self.read_json_body()
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        api_key = str(payload.get("apiKey", "")).strip()
        if not api_key:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing Poe API key."})
            return

        try:
            balance = fetch_current_balance(api_key)
            record_balance_snapshot(None, "manual", balance["balance"], balance["payload"])
            sync_result = sync_usage_history(
                api_key,
                pages=int(payload.get("pages") or DEFAULT_USAGE_SYNC_PAGES),
                limit=min(int(payload.get("limit") or DEFAULT_USAGE_SYNC_LIMIT), 100),
            )
        except urllib.error.HTTPError as error:
            parsed_error = extract_error_payload(error.read().decode("utf-8", errors="replace"))
            self.send_json(
                error.code,
                {
                    "error": parsed_error.get("message"),
                    "type": parsed_error.get("type"),
                    "code": parsed_error.get("code"),
                },
            )
            return
        except urllib.error.URLError as error:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Unable to reach Poe API: {error.reason}"})
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "balance": balance["balance"],
                "sync": sync_result,
                "summary": audit_summary(),
            },
        )

    def handle_create_folder(self) -> None:
        try:
            payload = self.read_json_body()
            folder = create_folder_record(payload.get("name"))
        except ValueError as error:
            status = HTTPStatus.CONFLICT if "已存在" in str(error) else HTTPStatus.BAD_REQUEST
            self.send_json(status, {"error": str(error)})
            return

        self.send_json(HTTPStatus.CREATED, {"folder": folder, "folders": folders_payload()})

    def handle_assign_conversation_folder(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 4 or parts[0] != "api" or parts[1] != "conversations" or parts[3] != "folder":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Conversation not found"})
            return

        try:
            payload = self.read_json_body()
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        folder_id = payload.get("folderId")
        if folder_id is not None:
            folder_id = str(folder_id).strip() or None

        try:
            conversation = assign_conversation_folder(
                conversation_id=parts[2],
                folder_id=folder_id,
                model=str(payload.get("model") or "").strip() or None,
                system_prompt=str(payload.get("systemPrompt") or "").strip() or None,
            )
        except LookupError as error:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
            return
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "conversation": conversation,
                "folders": folders_payload(),
                "conversations": conversation_library(),
            },
        )

    def handle_cancel(self, path: str) -> None:
        request_id = path.split("/")[3]
        request_row = load_request(request_id)
        if not request_row:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Request not found"})
            return

        update_request(request_id, cancel_requested=1)
        active_request = get_active_request(request_id)
        if active_request:
            active_request.cancel_event.set()
            add_request_event(request_id, "local.cancel_requested", {"requestId": request_id})

        self.send_json(HTTPStatus.OK, {"requestId": request_id, "status": "cancel_requested"})

    def handle_evidence_download(self, path: str) -> None:
        request_id = path.split("/")[3]
        try:
            archive_bytes = build_evidence_zip(request_id)
        except FileNotFoundError:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Request not found"})
            return

        filename = f"poe-evidence-{request_id}.zip"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(archive_bytes)))
        self.end_headers()
        self.wfile.write(archive_bytes)

    def handle_delete_request(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3 or parts[0] != "api" or parts[1] != "requests":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Request not found"})
            return

        request_id = parts[2]
        if get_active_request(request_id):
            self.send_json(HTTPStatus.CONFLICT, {"error": "Active request cannot be deleted."})
            return

        deleted_row = delete_request_record(request_id)
        if not deleted_row:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Request not found"})
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "requestId": request_id,
                "conversationId": deleted_row["conversation_id"],
                "deleted": True,
            },
        )

    def handle_delete_folder(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3 or parts[0] != "api" or parts[1] != "folders":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Folder not found"})
            return

        deleted_folder = delete_folder_record(parts[2])
        if not deleted_folder:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Folder not found"})
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "folder": deleted_folder,
                "folders": folders_payload(),
                "conversations": conversation_library(),
            },
        )

    def finalize_request(
        self,
        request_id: str,
        conversation_id: str,
        api_key: str,
        model: str,
        transport: str,
        client_state: dict[str, bool],
    ) -> dict[str, Any]:
        balance_after = None
        balance_error = None
        sync_error = None

        try:
            balance_result = fetch_current_balance(api_key)
            balance_after = balance_result["balance"]
            record_balance_snapshot(request_id, "after", balance_after, balance_result["payload"])
            self.safe_send_sse("balance", {"phase": "after", "currentPointBalance": balance_after}, client_state)
        except urllib.error.HTTPError as error:
            balance_error = extract_error_payload(error.read().decode("utf-8", errors="replace")).get("message")
        except urllib.error.URLError as error:
            balance_error = str(error.reason)

        try:
            sync_usage_history(api_key)
        except urllib.error.HTTPError as error:
            sync_error = extract_error_payload(error.read().decode("utf-8", errors="replace")).get("message")
        except urllib.error.URLError as error:
            sync_error = str(error.reason)

        request_row = load_request(request_id)
        if request_row and request_row["provider_response_id"] and transport == "responses":
            connection = db_connect()
            with connection:
                connection.execute(
                    """
                    UPDATE conversations
                    SET latest_response_id = ?, latest_transport = ?, updated_at = ?, model = ?
                    WHERE id = ?
                    """,
                    (request_row["provider_response_id"], transport, utc_ms(), model, conversation_id),
                )
            connection.close()

        request_row = load_request(request_id)
        return {
            "requestId": request_id,
            "status": request_row["status"] if request_row else None,
            "reconcileStatus": request_row["reconcile_status"] if request_row else None,
            "reconcileNote": request_row["reconcile_note"] if request_row else None,
            "assistantText": request_row["assistant_text"] if request_row else "",
            "assistantReasoningText": request_row["assistant_reasoning_text"] if request_row else "",
            "balanceAfter": request_row["ended_balance"] if request_row else balance_after,
            "balanceDelta": request_row["balance_delta"] if request_row else None,
            "usageCostPoints": request_row["usage_cost_points"] if request_row else None,
            "usageQueryId": request_row["usage_query_id"] if request_row else None,
            "balanceError": balance_error,
            "syncError": sync_error,
        }

    def handle_chat(self) -> None:
        try:
            payload = self.read_json_body()
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        api_key = str(payload.get("apiKey", "")).strip()
        conversation_id = str(payload.get("conversationId", "")).strip()
        model = str(payload.get("model", GPT54_PRO_MODEL_ID)).strip()
        system_prompt = str(payload.get("systemPrompt", "")).strip()
        min_balance_guard = payload.get("minBalanceGuard")
        temperature = payload.get("temperature")
        max_output_tokens = payload.get("maxOutputTokens")
        extra_body = payload.get("extraBody") if isinstance(payload.get("extraBody"), dict) else None
        extra_body = apply_model_parameter_defaults(model, extra_body)

        if not api_key:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing Poe API key."})
            return

        if not conversation_id:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing conversationId."})
            return

        if not model:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing model name."})
            return

        try:
            user_content = normalize_content(payload.get("userContent", ""))
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        transport = response_transport_for_content(user_content)
        previous_response_id = latest_response_id(conversation_id, model) if transport == "responses" else None

        ensure_conversation(conversation_id, model, system_prompt)

        request_summary = {
            "model": model,
            "transport": transport,
            "systemPrompt": system_prompt,
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "previousResponseId": previous_response_id,
            "attachmentManifest": attachment_manifest(user_content),
            "historyWindow": CHAT_HISTORY_WINDOW if transport == "chat_completions" else None,
            "notes": "Responses API is used for plain-text turns; chat.completions is used when attachments are present.",
        }

        request_id = create_request_record(
            conversation_id=conversation_id,
            model=model,
            transport=transport,
            system_prompt=system_prompt,
            user_content=user_content,
            request_summary=request_summary,
            previous_response_id=previous_response_id,
        )
        active_request = register_active_request(request_id)
        add_request_event(request_id, "local.request_created", {"transport": transport, "conversationId": conversation_id})

        try:
            balance_before = fetch_current_balance(api_key)
            record_balance_snapshot(request_id, "before", balance_before["balance"], balance_before["payload"])
        except urllib.error.HTTPError as error:
            parsed_error = extract_error_payload(error.read().decode("utf-8", errors="replace"))
            update_request(
                request_id,
                status="failed",
                reconcile_status="pending",
                reconcile_note="请求未发出，余额检查失败",
                error_type=parsed_error.get("type"),
                error_message=parsed_error.get("message"),
                http_status=error.code,
            )
            unregister_active_request(request_id)
            self.send_json(error.code, {"error": parsed_error.get("message"), "requestId": request_id})
            return
        except urllib.error.URLError as error:
            update_request(
                request_id,
                status="failed",
                reconcile_status="pending",
                reconcile_note="请求未发出，余额检查失败",
                error_type="network_error",
                error_message=f"Unable to reach Poe API: {error.reason}",
                http_status=502,
            )
            unregister_active_request(request_id)
            self.send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": f"Unable to reach Poe API: {error.reason}", "requestId": request_id},
            )
            return

        if isinstance(min_balance_guard, int) and balance_before["balance"] is not None:
            if balance_before["balance"] < min_balance_guard:
                update_request(
                    request_id,
                    status="failed",
                    reconcile_status="pending",
                    reconcile_note="触发最小剩余点数保护，未发送请求",
                    error_type="balance_guard",
                    error_message=f"Current balance {balance_before['balance']} is below the guard {min_balance_guard}.",
                    http_status=400,
                )
                unregister_active_request(request_id)
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "当前积分低于最小剩余点数保护阈值，已阻止发送。",
                        "requestId": request_id,
                        "currentPointBalance": balance_before["balance"],
                    },
                )
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        client_state = {"connected": True}
        self.safe_send_sse(
            "ready",
            {
                "requestId": request_id,
                "conversationId": conversation_id,
                "model": model,
                "transport": transport,
                "currentPointBalance": balance_before["balance"],
            },
            client_state,
        )

        response = None
        saw_visible_text = False
        saw_saved_output = False

        try:
            if transport == "responses":
                request_body: dict[str, Any] = {
                    "model": model,
                    "input": content_to_text(user_content),
                    "instructions": system_prompt or None,
                    "stream": True,
                    "store": True,
                    "metadata": {
                        "local_request_id": request_id,
                        "conversation_id": conversation_id,
                    },
                }
                if previous_response_id:
                    request_body["previous_response_id"] = previous_response_id
                if isinstance(temperature, (int, float)):
                    request_body["temperature"] = temperature
                if isinstance(max_output_tokens, int):
                    request_body["max_output_tokens"] = max_output_tokens
                if extra_body:
                    request_body.update(extra_body)
                request = urllib.request.Request(
                    url=f"{POE_BASE_URL}/responses",
                    data=json_dumps(request_body).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/event-stream",
                    },
                    method="POST",
                )
            else:
                request_body = {
                    "model": model,
                    "messages": build_chat_history_messages(conversation_id, user_content, system_prompt),
                    "stream": True,
                }
                if isinstance(temperature, (int, float)):
                    request_body["temperature"] = temperature
                if isinstance(max_output_tokens, int):
                    request_body["max_completion_tokens"] = max_output_tokens
                if extra_body:
                    request_body["extra_body"] = extra_body
                request = urllib.request.Request(
                    url=f"{POE_BASE_URL}/chat/completions",
                    data=json_dumps(request_body).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/event-stream",
                    },
                    method="POST",
                )

            response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
            update_request(request_id, status="streaming", **rate_limit_fields_from_headers(response.headers))

            for data in iter_sse_data_lines(response):
                if active_request.cancel_event.is_set():
                    add_request_event(request_id, "local.cancel_applied", {"requestId": request_id})
                    raise RuntimeError("__cancelled__")

                if data == "[DONE]":
                    add_request_event(request_id, "provider.done")
                    update_request(request_id, had_done=1)
                    self.safe_send_sse("status", {"status": "done-marker", "requestId": request_id}, client_state)
                    break

                parsed = json.loads(data)

                if transport == "responses":
                    event_type = parsed.get("type")
                    add_request_event(request_id, event_type or "responses.unknown", parsed)

                    if event_type == "response.created":
                        response_object = parsed.get("response") or {}
                        if response_object.get("id"):
                            update_request(request_id, provider_response_id=response_object["id"])
                        self.safe_send_sse("status", {"status": "streaming", "requestId": request_id}, client_state)
                        continue

                    if event_type == "response.output_text.delta":
                        delta = str(parsed.get("delta", ""))
                        if delta:
                            saw_visible_text = True
                            saw_saved_output = True
                            append_request_output(request_id, delta)
                            self.safe_send_sse("delta", {"content": delta, "requestId": request_id}, client_state)
                        continue

                    if event_type == "response.completed":
                        response_object = parsed.get("response") or {}
                        usage_payload = response_object.get("usage") or {}
                        completed_text = extract_response_output_text(response_object)
                        if completed_text and not saw_visible_text:
                            saw_visible_text = True
                            saw_saved_output = True
                            append_request_output(request_id, completed_text)
                            self.safe_send_sse("delta", {"content": completed_text, "requestId": request_id}, client_state)
                        final_row = load_request(request_id)
                        final_status = (
                            "completed"
                            if final_row and str(final_row["assistant_text"] or "").strip()
                            else ("partial_saved" if request_has_saved_output(final_row) or saw_saved_output else "failed")
                        )
                        update_request(
                            request_id,
                            status=final_status,
                            provider_response_id=response_object.get("id"),
                            assistant_label=response_object.get("model", model),
                            response_meta_json=json_dumps({"status": response_object.get("status"), "model": response_object.get("model", model)}),
                            had_done=1,
                            input_tokens=usage_payload.get("input_tokens"),
                            output_tokens=usage_payload.get("output_tokens"),
                            total_tokens=usage_payload.get("total_tokens"),
                        )
                        self.safe_send_sse("usage", usage_payload, client_state)
                        self.safe_send_sse(
                            "status",
                            {"status": final_status, "requestId": request_id},
                            client_state,
                        )
                        break

                    if event_type in {"response.failed", "response.incomplete"}:
                        response_object = parsed.get("response") or {}
                        update_request(
                            request_id,
                            status="partial_saved" if saw_saved_output else "failed",
                            provider_response_id=response_object.get("id"),
                            response_meta_json=json_dumps({"status": response_object.get("status")}),
                            had_done=0,
                            assistant_label=model,
                        )
                        break

                else:
                    add_request_event(request_id, "chat.chunk", parsed)
                    provider_error = parsed.get("error") if isinstance(parsed, dict) else None
                    if isinstance(provider_error, dict):
                        status = "partial_saved" if saw_saved_output else "failed"
                        update_request(
                            request_id,
                            status=status,
                            error_type=provider_error.get("type"),
                            error_message=provider_error.get("message"),
                            assistant_label=model,
                        )
                        self.safe_send_sse(
                            "error",
                            {
                                "requestId": request_id,
                                "message": provider_error.get("message") or "Provider returned an error chunk.",
                                "status": status,
                            },
                            client_state,
                        )
                        break
                    choice = (parsed.get("choices") or [{}])[0]
                    delta_payload = choice.get("delta") or {}
                    delta = str(delta_payload.get("content") or "")
                    reasoning_delta = str(delta_payload.get("reasoning_content") or "")
                    finish_reason = choice.get("finish_reason")
                    if delta:
                        saw_visible_text = True
                        saw_saved_output = True
                        append_request_output(request_id, delta)
                        self.safe_send_sse("delta", {"content": delta, "requestId": request_id}, client_state)
                    if reasoning_delta:
                        saw_saved_output = True
                        append_request_reasoning(request_id, reasoning_delta)
                        self.safe_send_sse(
                            "reasoning",
                            {"content": reasoning_delta, "requestId": request_id},
                            client_state,
                        )
                    if finish_reason:
                        final_row = load_request(request_id)
                        final_status = (
                            "completed"
                            if final_row and str(final_row["assistant_text"] or "").strip()
                            else ("partial_saved" if request_has_saved_output(final_row) or saw_saved_output else "failed")
                        )
                        update_request(request_id, status=final_status, had_done=1, assistant_label=model)
                        self.safe_send_sse(
                            "status",
                            {"status": final_status, "finishReason": finish_reason, "requestId": request_id},
                            client_state,
                        )
                        break

            current_row = load_request(request_id)
            if current_row and current_row["status"] == "streaming":
                update_request(
                    request_id,
                    status=(
                        "completed"
                        if current_row["had_done"] and str(current_row["assistant_text"] or "").strip()
                        else ("partial_saved" if request_has_saved_output(current_row) or saw_saved_output else "failed")
                    ),
                    assistant_label=model,
                )

        except urllib.error.HTTPError as error:
            parsed_error = extract_error_payload(error.read().decode("utf-8", errors="replace"))
            status = "partial_saved" if saw_saved_output else "failed"
            update_request(
                request_id,
                status=status,
                error_type=parsed_error.get("type"),
                error_message=parsed_error.get("message"),
                http_status=error.code,
                **rate_limit_fields_from_headers(error.headers),
            )
            self.safe_send_sse(
                "error",
                {"requestId": request_id, "message": parsed_error.get("message"), "status": status},
                client_state,
            )
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", error)
            status = "timed_out" if isinstance(reason, socket.timeout) else ("partial_saved" if saw_saved_output else "failed")
            update_request(
                request_id,
                status=status,
                error_type="timeout_error" if isinstance(reason, socket.timeout) else "network_error",
                error_message=str(reason),
                http_status=504 if isinstance(reason, socket.timeout) else 502,
            )
            self.safe_send_sse(
                "error",
                {"requestId": request_id, "message": str(reason), "status": status},
                client_state,
            )
        except RuntimeError as error:
            if str(error) == "__cancelled__":
                current_row = load_request(request_id)
                partial_text = current_row["assistant_text"] if current_row else ""
                partial_reasoning = current_row["assistant_reasoning_text"] if current_row else ""
                update_request(
                    request_id,
                    status="partial_saved" if (partial_text or partial_reasoning) else "failed",
                    error_type="cancelled_by_user",
                    error_message="Generation was cancelled from the local GUI.",
                    http_status=499,
                )
                self.safe_send_sse(
                    "cancelled",
                    {"requestId": request_id, "status": "partial_saved" if (partial_text or partial_reasoning) else "failed"},
                    client_state,
                )
            else:
                update_request(
                    request_id,
                    status="failed",
                    error_type="runtime_error",
                    error_message=str(error),
                    http_status=500,
                )
                self.safe_send_sse(
                    "error",
                    {"requestId": request_id, "message": str(error), "status": "failed"},
                    client_state,
                )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

            finalize_payload = self.finalize_request(
                request_id=request_id,
                conversation_id=conversation_id,
                api_key=api_key,
                model=model,
                transport=transport,
                client_state=client_state,
            )
            final_row = load_request(request_id)
            self.safe_send_sse(
                "completed",
                {
                    **finalize_payload,
                    "assistantText": final_row["assistant_text"] if final_row else "",
                    "assistantReasoningText": final_row["assistant_reasoning_text"] if final_row else "",
                    "assistantLabel": final_row["assistant_label"] if final_row else model,
                },
                client_state,
            )
            unregister_active_request(request_id)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    init_db()
    server = ReusableThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"poe-gpt-5-4-pro-chat listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
