from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from lab5_config import LAB5_HISTORY_PATH, ensure_lab5_dirs


def save_tool_event(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    source: str = "bot",
) -> None:
    ensure_lab5_dirs()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
    }
    with LAB5_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_tool_history(limit: int = 20) -> list[dict[str, Any]]:
    if not LAB5_HISTORY_PATH.exists():
        return []
    lines = LAB5_HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records

