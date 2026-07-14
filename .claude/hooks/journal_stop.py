#!/usr/bin/env python3
"""Stop hook: nudge Claude to log a journal.md entry once per session before finishing."""
import json
import os
import sys

MARKER_DIR = "/tmp/claude-journal-hook"


def main() -> None:
    data = json.load(sys.stdin)
    session_id = data.get("session_id", "unknown")

    os.makedirs(MARKER_DIR, exist_ok=True)
    marker = os.path.join(MARKER_DIR, f"{session_id}.done")

    if os.path.exists(marker):
        print(json.dumps({}))
        return

    open(marker, "w").close()
    print(json.dumps({
        "decision": "block",
        "reason": (
            "세션을 끝내기 전에 docs/journal.md에 이번 세션 작업 요약을 기록해줘. "
            "형식: '## YYYY-MM-DD (N차)' 헤더(오늘 날짜의 기존 항목 수 + 1을 N차로 사용) "
            "아래에 핵심 변경/결정/다음 할 일을 불릿 포인트로 적고, 기록한 뒤 다시 끝내줘."
        ),
    }))


if __name__ == "__main__":
    main()
