"""
B3 Scheduler 로그 분석기 v2.0
- GPT-5.2: BOM/인코딩 자동 감지
- Gemini: 3단계 fallback 파싱
- DeepSeek: 상세 디버그 출력
"""

from __future__ import annotations

import ast
import re
import sys
from typing import Dict, Optional, Tuple


def read_text_auto(path: str) -> Tuple[str, str]:
    raw = open(path, "rb").read()

    enc: Optional[str] = None
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        enc = "utf-16"
    elif raw.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    elif b"\x00" in raw[:4096]:
        enc = "utf-16"

    for candidate in [enc, "utf-8", "utf-8-sig", "utf-16", "cp949", "latin-1"]:
        if not candidate:
            continue
        try:
            text = raw.decode(candidate, errors="replace")
            text = text.replace("\x00", "")
            return text, candidate
        except Exception:
            continue

    return raw.decode("utf-8", errors="replace"), "utf-8 (fallback)"


def parse_actions_from_summary(text: str) -> Optional[Dict[str, int]]:
    patterns = [
        r"Actions:\s*(\{[^}\n\r]+\})",
        r"\|\s*Actions:\s*(\{[^}\n\r]+\})",
        r"\[Summary\].*?Actions:\s*(\{[^}\n\r]+\})",
    ]

    for line in reversed(text.splitlines()):
        if "Actions:" not in line:
            continue
        for pat in patterns:
            m = re.search(pat, line)
            if not m:
                continue
            try:
                obj = ast.literal_eval(m.group(1))
                if isinstance(obj, dict):
                    return {str(k): int(v) for k, v in obj.items()}
            except Exception:
                continue
    return None


def parse_actions_from_result(text: str) -> Optional[Dict[str, int]]:
    m = re.search(r"^Result:\s*(\{.*\})\s*$", text, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return None
    try:
        result = ast.literal_eval(m.group(1))
        if not isinstance(result, dict):
            return None

        actions: Dict[str, int] = {}
        for ex_data in result.values():
            if not isinstance(ex_data, dict):
                continue
            for v in ex_data.values():
                a = (v or {}).get("action", "UNKNOWN")
                actions[str(a)] = actions.get(str(a), 0) + 1
        return actions if actions else None
    except Exception:
        return None


def parse_actions_regex_fallback(text: str) -> Optional[Dict[str, int]]:
    actions = {}
    patterns = {
        "BUY": r"'BUY':\s*(\d+)",
        "SELL": r"'SELL':\s*(\d+)",
        "HOLD": r"'HOLD':\s*(\d+)",
        "SKIP": r"'SKIP':\s*(\d+)",
        "ERROR": r"'ERROR':\s*(\d+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            actions[key] = int(m.group(1))

    return actions if actions else None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python b3_analyze.py <path_to_scheduler_log>")
        return 2

    path = argv[1]
    print(f"[DEBUG] 분석 시작: {path}")

    text, enc = read_text_auto(path)
    print(f"[DEBUG] 인코딩: {enc}, 파일 크기: {len(text)} 문자")

    actions = parse_actions_from_summary(text)
    source = "summary"

    if actions is None:
        print("[DEBUG] Summary 파싱 실패, Result 시도")
        actions = parse_actions_from_result(text)
        source = "result"

    if actions is None:
        print("[DEBUG] Result 파싱 실패, 정규식 fallback 시도")
        actions = parse_actions_regex_fallback(text)
        source = "regex_fallback"

    if actions is None:
        print("B3 결과: FAIL (could not parse Actions/Result)")
        print(f"[DEBUG] 마지막 20줄:\n" + "\n".join(text.splitlines()[-20:]))
        return 1

    buy = int(actions.get("BUY", 0))
    sell = int(actions.get("SELL", 0))
    hold = int(actions.get("HOLD", 0))
    skip = int(actions.get("SKIP", 0))
    err = int(actions.get("ERROR", 0))

    print(
        f"BUY={buy} SELL={sell} HOLD={hold} SKIP={skip} ERROR={err} (source={source})"
    )

    if err != 0:
        print(f"B3 결과: FAIL (ERROR={err})")
        return 1

    if buy == 0 and sell == 0:
        print("B3 결과: PASS_WITH_WARN (ERROR=0 but BUY/SELL=0; 시장상황 가능)")
        return 0

    print(f"B3 결과: PASS (ERROR=0, BUY={buy}, SELL={sell}, HOLD={hold}, SKIP={skip})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
