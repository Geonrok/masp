import ast, re, sys

path = sys.argv[1]
txt = open(path, "r", encoding="utf-8", errors="replace").read()

# 1) Summary Actions 파싱
m = re.search(r"Actions:\s*(\{[^\n\r]*\})", txt)
actions = None
if m:
    try:
        actions = ast.literal_eval(m.group(1))
    except Exception:
        actions = None

# 2) 대안: Result: {...} 파싱
if actions is None:
    m2 = re.search(r"^Result:\s*(\{.*\})\s*$", txt, flags=re.MULTILINE)
    if m2:
        try:
            result = ast.literal_eval(m2.group(1))
            actions = {}
            up = result.get("upbit", {})
            for _, v in up.items():
                a = (v or {}).get("action", "UNKNOWN")
                actions[a] = actions.get(a, 0) + 1
        except Exception:
            actions = None

if actions is None:
    print("B3 결과: FAIL (could not parse Actions/Result)")
    sys.exit(1)

buy = int(actions.get("BUY", 0))
sell = int(actions.get("SELL", 0))
hold = int(actions.get("HOLD", 0))
err = int(actions.get("ERROR", 0))

print(f"BUY={buy} SELL={sell} HOLD={hold} ERROR={err}")

if err != 0:
    print(f"B3 결과: FAIL (ERROR={err})")
    sys.exit(1)

if buy == 0 and sell == 0:
    print(f"B3 결과: PASS_WITH_WARN (ERROR=0 but BUY/SELL=0; 시장상황 가능)")
else:
    print(f"B3 결과: PASS (ERROR=0, BUY={buy}, SELL={sell}, HOLD={hold})")
