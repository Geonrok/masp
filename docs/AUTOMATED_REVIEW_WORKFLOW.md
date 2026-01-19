# Automated Code Review Workflow

MASP í”„ë¡œì íŠ¸ì˜ ìë™í™”ëœ ì½”ë“œ ë¦¬ë·° ì›Œí¬í”Œë¡œìš° ê°€ì´ë“œ.

## Overview

ì½”ë“œ ì‘ì„± í›„ ì»¤ë°‹ ì „ `codex review --uncommitted` ëª…ë ¹ìœ¼ë¡œ ìë™ ì½”ë“œ ë¦¬ë·°ë¥¼ ìˆ˜í–‰í•©ë‹ˆë‹¤.

## Workflow

```
Code â†’ __init__.py Export â†’ pytest â†’ codex review â†’ Fix Issues â†’ Commit
```

### Step-by-Step

1. **ì½”ë“œ ì‘ì„±**
   - ìƒˆ ì»´í¬ë„ŒíŠ¸/ê¸°ëŠ¥ êµ¬í˜„
   - ê¸°ì¡´ ì½”ë“œ ìˆ˜ì •

2. **Export ì¶”ê°€** (í•´ë‹¹ ì‹œ)
   ```python
   # services/dashboard/components/__init__.py
   from services.dashboard.components.new_component import render_new_component
   ```

3. **í…ŒìŠ¤íŠ¸ ì‹¤í–‰**
   ```bash
   python -m pytest tests/dashboard/test_new_component.py -v --tb=short
   ```

4. **Codex Review**
   ```bash
   cd "E:\íˆ¬ì\Multi-Asset Strategy Platform"
   codex review --uncommitted
   ```

5. **ì´ìŠˆ ìˆ˜ì •** (ë°œê²¬ ì‹œ)
   - P1 (Critical): ì¦‰ì‹œ ìˆ˜ì • í•„ìˆ˜
   - P2 (Important): ìˆ˜ì • ê¶Œì¥
   - P3 (Minor): ì„ íƒì  ìˆ˜ì •

6. **ì»¤ë°‹**
   ```bash
   git add .
   git commit -m "Add feature X"
   ```

## Review Application Policy

| ë¶„ë¥˜ | ì ìš© ê¸°ì¤€ |
|------|----------|
| **í•„ìˆ˜ë³´ê°•** (P1) | í•­ìƒ ì ìš© |
| **ê¶Œì¥ë³´ê°•** (P2) | í•­ìƒ ì ìš© |
| **ì„ íƒë³´ê°•** (P3) | ëª…ëª…/ìŠ¤íƒ€ì¼ë§Œ ìŠ¤í‚µ |

**ì´ìœ **: ì™„ì„±ë„ ìš°ì„ . ë‚˜ì¤‘ì— ë‹¤ì‹œ ì†ëŒ€ì§€ ì•Šê¸° ìœ„í•´.

## Priority Levels

### P1 - Critical (í•„ìˆ˜ë³´ê°•)
- ë³´ì•ˆ ì·¨ì•½ì 
- ë°ì´í„° ì†ì‹¤ ìœ„í—˜
- ëŸ°íƒ€ì„ ì—ëŸ¬ ê°€ëŠ¥ì„±
- ì‹¬ê°í•œ ë¡œì§ ì˜¤ë¥˜

**ì˜ˆì‹œ:**
- SQL Injection ì·¨ì•½ì 
- ì¸ì¦/ì¸ê°€ ëˆ„ë½
- Null pointer dereference
- ë¬´í•œ ë£¨í”„ ê°€ëŠ¥ì„±

### P2 - Important (ê¶Œì¥ë³´ê°•)
- ì ì¬ì  ë²„ê·¸
- ì„±ëŠ¥ ì´ìŠˆ
- ì—£ì§€ ì¼€ì´ìŠ¤ ë¯¸ì²˜ë¦¬
- API ì„¤ê³„ ë¬¸ì œ

**ì˜ˆì‹œ:**
- Timezone ì²˜ë¦¬ ì˜¤ë¥˜
- ë¹ˆ ë¦¬ìŠ¤íŠ¸/None ì²˜ë¦¬ ëˆ„ë½
- ë¹„íš¨ìœ¨ì  ì•Œê³ ë¦¬ì¦˜
- ë¶€ì ì ˆí•œ ì—ëŸ¬ í•¸ë“¤ë§

### P3 - Minor (ì„ íƒë³´ê°•)
- ì½”ë“œ ìŠ¤íƒ€ì¼
- ë„¤ì´ë° ì»¨ë²¤ì…˜
- ë¬¸ì„œí™” ë¶€ì¡±
- ë¦¬íŒ©í† ë§ ì œì•ˆ

**ì˜ˆì‹œ:**
- ë³€ìˆ˜ëª… ê°œì„  ì œì•ˆ
- ì£¼ì„ ì¶”ê°€ ê¶Œì¥
- í•¨ìˆ˜ ë¶„ë¦¬ ì œì•ˆ
- íƒ€ì… íŒíŠ¸ ì¶”ê°€

## Common Patterns

### Datetime Timezone Handling

```python
# Bad: TypeError when comparing mixed tz-aware/naive
if dt1 < dt2:  # Raises if mixed

# Good: Safe comparison with fallback
def _safe_datetime_compare(dt1, dt2):
    try:
        if dt1 < dt2: return -1
        elif dt1 > dt2: return 1
        return 0
    except TypeError:
        # Fallback: strip tzinfo
        dt1_naive = dt1.replace(tzinfo=None) if dt1.tzinfo else dt1
        dt2_naive = dt2.replace(tzinfo=None) if dt2.tzinfo else dt2
        ...
```

### Percentage Formatting

```python
# For returns/PnL (signed)
def _format_percent(value):
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"  # +15.50%, -8.25%

# For MDD/Volatility/Win Rate (unsigned)
def _format_plain_percent(value):
    return f"{value:.2f}%"  # 22.50%, 8.50%
```

### Filter Fallback Pattern

```python
def _filter_by_criteria(items, criteria, allow_fallback=True):
    filtered = [item for item in items if matches(item, criteria)]

    if filtered:
        return filtered, False  # (result, used_fallback)
    elif allow_fallback:
        return items, True  # Return all with warning flag
    else:
        return [], False  # Return empty
```

### Deterministic Demo Data

```python
# Use fixed reference date for demo mode
_DEMO_REFERENCE_DATE = datetime(2026, 1, 1, 12, 0, 0)

def _get_demo_data():
    base_time = _DEMO_REFERENCE_DATE
    return [
        Item(timestamp=base_time - timedelta(minutes=1), ...),
        Item(timestamp=base_time - timedelta(minutes=5), ...),
    ]
```

## Troubleshooting

### Review ê²°ê³¼ê°€ ë¹„ì–´ìˆì„ ë•Œ
- ë³€ê²½ ì‚¬í•­ì´ staging ë˜ì–´ ìˆëŠ”ì§€ í™•ì¸
- `git status` ë¡œ modified íŒŒì¼ í™•ì¸

### P2 ì´ìŠˆê°€ ê³„ì† ë°œìƒí•  ë•Œ
- ê·¼ë³¸ ì›ì¸ ë¶„ì„ (edge case ëˆ„ë½ ë“±)
- í…ŒìŠ¤íŠ¸ ì¼€ì´ìŠ¤ ì¶”ê°€ë¡œ ì»¤ë²„ë¦¬ì§€ í™•ë³´

### codex ëª…ë ¹ ì‹¤íŒ¨ ì‹œ
- API í‚¤ ì„¤ì • í™•ì¸
- ë„¤íŠ¸ì›Œí¬ ì—°ê²° í™•ì¸
- `codex --version` ìœ¼ë¡œ ì„¤ì¹˜ í™•ì¸

## Integration with CI/CD

```yaml
# .github/workflows/review.yml (ì˜ˆì‹œ)
name: Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Codex Review
        run: codex review --uncommitted
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## References

- [Codex CLI Documentation](https://github.com/openai/codex)
- [MASP Coding Checklist](./CODING_CHECKLIST.md)
# MASP ?ë™ ê²€???Œí¬?Œë¡œ??
## ê°œìš”
Claude Code Opus 4.5?€ Codex CLIë¥??°ë™???ë™ ì½”ë“œ ë¦¬ë·° ?œìŠ¤??
## ?„í‚¤?ì²˜
```
?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€???? Claude Code Opus 4.5               ???? - ê³„íš ?˜ë¦½                         ???? - ì½”ë“œ ?‘ì„±                         ???? - ?ŒìŠ¤???‘ì„±                       ???”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?¬â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??                  ???ë™ ?¸ì¶œ
                  ???Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€???? codex review --uncommitted         ???? - ë¹„ë??”í˜• ì½”ë“œ ë¦¬ë·°               ???? - ?„ìˆ˜/ê¶Œì¥/? íƒ ë³´ê°• ì¶œë ¥          ???”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?¬â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??                  ??ê²°ê³¼ ë¶„ì„
                  ???Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€???? Claude Code                        ???? - ê²€??ê²°ê³¼ ?ìš©                    ???? - ?¬í…Œ?¤íŠ¸                         ???? - ?µê³¼ ??ì»¤ë°‹                     ???”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??```

## ?¬ì „ ?”êµ¬?¬í•­

### 1. Claude Code ?¤ì¹˜
```powershell
npm install -g @anthropic-ai/claude-code
```

### 2. Codex CLI ?¤ì¹˜
```powershell
npm install -g @openai/codex
```

### 3. ?¸ì¦ ?¤ì •
```powershell
# Claude Code OAuth ? í° (?êµ¬ ?¤ì •)
[Environment]::SetEnvironmentVariable("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-...", "User")

# Codex??ë³„ë„ ë¡œê·¸??codex login
```

## ?Œí¬?Œë¡œ???ì„¸

### Phase 1: ì½”ë“œ ?‘ì„±
1. Claude Codeê°€ ?”êµ¬?¬í•­ ë¶„ì„
2. ì»´í¬?ŒíŠ¸ ì½”ë“œ ?‘ì„± (?? `component.py`)
3. ?ŒìŠ¤??ì½”ë“œ ?‘ì„± (?? `test_component.py`)
4. `__init__.py` export ì¶”ê?

### Phase 2: ë¡œì»¬ ?ŒìŠ¤??```powershell
pytest tests/dashboard/test_component.py -v
pytest --tb=short -q  # ?„ì²´ ?ŒìŠ¤??```

### Phase 3: ?ë™ ê²€??```powershell
cd "E:\?¬ì\Multi-Asset Strategy Platform"
codex review --uncommitted
```

### Phase 4: ê²€??ê²°ê³¼ ?ìš©

#### ?ìš© ?ì¹™
| êµ¬ë¶„ | ?ìš© ?¬ë? | ê¸°ì? |
|------|----------|------|
| ?„ìˆ˜ë³´ê°• | ????ƒ | - |
| ê¶Œì¥ë³´ê°• | ???ìš© | ?°í????ˆì™¸, ?°ì´???¤ì—¼, UX ?¼ê??? ë³´ì•ˆ, ë©”ëª¨ë¦??„ìˆ˜, ?ëŸ¬ ?¸ë“¤ë§?|
| ê¶Œì¥ë³´ê°• | ???¤í‚µ | ëª…ëª… ë³€ê²? ì½”ë“œ ?•ë¦¬, ?¤í??? ?±ëŠ¥ ìµœì ??(ë³‘ëª© ?„ë‹Œ ê²½ìš°) |

#### ?ë‹¨ ê¸°ì?
> "?„ë¡œ?•ì…˜?ì„œ ?¥ì• ???ì‹¤??ë°œìƒ?????ˆëŠ”ê°€?"

### Phase 5: ì»¤ë°‹
```powershell
git add <files>
git commit -m "Add component description"
```

## Claude Code ?„ë¡¬?„íŠ¸ ?œí”Œë¦?
### ??ì»´í¬?ŒíŠ¸ ?‘ì—… ?œì‘
```
## [Phase]-[ë²ˆí˜¸] ?‘ì—…: [ì»´í¬?ŒíŠ¸ëª?

### ?Œí¬?Œë¡œ??(?„ì²´ ?ë™??
1. ì½”ë“œ ?‘ì„± ???Œì¼ ?€??2. __init__.py??export ì¶”ê?
3. pytest ?¤í–‰ ???„ì²´ ?µê³¼ ?•ì¸
4. codex review --uncommitted ?¤í–‰
5. ê²€??ê²°ê³¼ ë¶„ì„ ë°??ìš©
6. ?¬í…Œ?¤íŠ¸ ???µê³¼ ??ì»¤ë°‹

### ê²€???ìš© ?ì¹™
- ?„ìˆ˜ë³´ê°•: ??ƒ ?ìš©
- ê¶Œì¥ë³´ê°•: ?°í????ˆì •??UX ê´€?¨ë§Œ ?ìš©
- ëª…ëª…/?¤í???ê´€?? ?¤í‚µ

?‘ì—… ?œì‘?´ì¤˜.
```

## ?¤í–‰ ?ˆì‹œ

### ?±ê³µ ì¼€?´ìŠ¤
```
Claude Code: ì½”ë“œ ?‘ì„± ?„ë£Œ, pytest 38 passed
Claude Code: codex review --uncommitted ?¤í–‰
Codex: ?µê³¼, ê¶Œì¥ë³´ê°• 2ê±?(timezone ê´€??
Claude Code: ê¶Œì¥ë³´ê°• ?ìš© (?°í????ˆì •??
Claude Code: pytest 42 passed
Claude Code: git commit "Add component"
```

### ?˜ì • ?„ìš” ì¼€?´ìŠ¤
```
Claude Code: ì½”ë“œ ?‘ì„± ?„ë£Œ, pytest 35 passed
Claude Code: codex review --uncommitted ?¤í–‰
Codex: ?„ìˆ˜ë³´ê°• 1ê±?(f-string ?¸ì½”??ê¹¨ì§)
Claude Code: ?„ìˆ˜ë³´ê°• ?ìš©
Claude Code: pytest 35 passed
Claude Code: codex review --uncommitted ?¬ì‹¤??Codex: ?µê³¼
Claude Code: git commit "Add component"
```

## ?¥ì 

1. **?ˆì§ˆ ?¥ìƒ**: ?¤ë¥¸ AI???œê°?¼ë¡œ ì½”ë“œ ê²€ì¦?2. **?ë™??*: ?˜ë™ ë³µì‚¬/ë¶™ì—¬?£ê¸° ?œê±°
3. **?¼ê???*: ?™ì¼??ê²€??ê¸°ì? ?ìš©
4. **?ë„**: ê²€??ë£¨í”„ ?ë™ ë°˜ë³µ

## ?œí•œ?¬í•­

1. Codex CLI ?¬ìš©?‰ì? GPT ?¹ê³¼ ?œë„ ê³µìœ 
2. `codex review`??uncommitted ë³€ê²½ì‚¬??§Œ ê²€??3. ë³µì¡???„í‚¤?ì²˜ ë¦¬ë·°???˜ë™ ê²€??ê¶Œì¥

## ê´€???Œì¼

- `E:\AI_Review\scripts\Extract-ReviewFiles.ps1` - ?˜ë™ ê²€?˜ìš© ?¤í¬ë¦½íŠ¸
- `E:\AI_Review\templates\GPT_REVIEW_TEMPLATE.md` - ê²€???œí”Œë¦?- `docs/CODING_CHECKLIST.md` - ì½”ë”© ì²´í¬ë¦¬ìŠ¤??
---

## ë³‘ë ¬ ?€??ë¦¬ë·° ?œìŠ¤??(v2.0)

### 1. ê°œìš”
Codex CLI?€ Gemini CLIë¥??™ì‹œ???¤í–‰?˜ì—¬ ?í˜¸ ë³´ì™„?ì¸ ì½”ë“œ ë¦¬ë·°ë¥??˜í–‰?©ë‹ˆ??

### 2. ?¬ìš©ë²??ˆì‹œ
```powershell
# ê¸°ë³¸ ?¤í–‰ (Uncommitted Changes)
.\scripts\review-parallel.ps1

# Staged ?Œì¼ ê²€??.\scripts\review-parallel.ps1 -Target --staged

# ?¹ì • ?Œì¼ ê²€??.\scripts\review-parallel.ps1 -Target "services/market_data.py" -Quiet
```

### 3. ì¶œë ¥ ?Œì¼ êµ¬ì¡°
```
review-results/
?œâ??€ codex_review_20260119_120000.md   (Codex ?ì„¸ ê²°ê³¼)
?œâ??€ gemini_review_20260119_120000.md  (Gemini ?ì„¸ ê²°ê³¼)
?”â??€ review_summary_20260119_120000.md (?µí•© ?”ì•½ ë¦¬í¬??
```

### 4. AIë³?ê²€??ì´ˆì 
| êµ¬ë¶„ | Codex | Gemini |
|------|-------|--------|
| **ì£?ê°•ì ** | êµ¬ë¬¸ ?•í™•?? ?¼ì´ë¸ŒëŸ¬ë¦??¬ìš©ë²?| ë³´ì•ˆ ì·¨ì•½?? ?„í‚¤?ì²˜, ?£ì? ì¼€?´ìŠ¤ |
| **ê²€???ì—­** | ?€???ŒíŠ¸, PEP 8, API ?¬ìš© | ë¹„ì¦ˆ?ˆìŠ¤ ë¡œì§, ?°ì´???ë¦„, ?ˆì™¸ ì²˜ë¦¬ |
| **?¼ë“œë°??¤í???* | êµ¬ì²´?ì¸ ì½”ë“œ ?œì•ˆ | ?˜ì´ ?ˆë²¨ ë¶„ì„ ë°?ë¦¬ìŠ¤??ê²½ê³  |

### 5. ?¼ë“œë°?ì¶©ëŒ ???°ì„ ?œìœ„ ê·œì¹™
1. **ë³´ì•ˆ ?´ìŠˆ:** Gemini ?˜ê²¬ ?°ì„  (ë³´ìˆ˜???‘ê·¼)
2. **ì½”ë“œ ?¤í???** Codex ?˜ê²¬ ?°ì„  (?œì? ì¤€??
3. **ë¡œì§/?±ëŠ¥:** ??AIê°€ ëª¨ë‘ ì§€?í•œ ê²½ìš° **?„ìˆ˜ ?˜ì • (P1)**

### 6. ?Œí¬?Œë¡œ???¤ì´?´ê·¸??```
                 ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??                 ?? Code Changes    ??                 ?”â??€?€?€?€?€?€?€?¬â??€?€?€?€?€?€?€?€??                          ??            ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?´â??€?€?€?€?€?€?€?€?€?€?€?€??            ??                          ??    ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€??          ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€??    ??Codex Review  ??          ??Gemini Review ??    ??(Syntax/API)  ??          ??(Security/Biz)??    ?”â??€?€?€?€?€?€?¬â??€?€?€?€?€?€??          ?”â??€?€?€?€?€?€?¬â??€?€?€?€?€?€??            ??                          ??            ?”â??€?€?€?€?€?€?€?€?€?€?€?€?¬â??€?€?€?€?€?€?€?€?€?€?€?€??                          ??                 ?Œâ??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??                 ?? Summary Report  ??                 ??(Merge Findings) ??                 ?”â??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€??```

### 7. ê¸°ë? ?¨ê³¼
- **?œê°„ ?¨ì¶•:** ë³‘ë ¬ ?¤í–‰?¼ë¡œ ë¦¬ë·° ?œê°„ 50% ?ˆê°
- **?ˆì§ˆ ?¥ìƒ:** ?í˜¸ ë³´ì™„?ì¸ ê´€??Syntax vs Logic)?¼ë¡œ ?¬ê°ì§€?€ ?œê±°
- **?ˆì •??ê°•í™”:** Gemini??ë³´ì•ˆ/?„í‚¤?ì²˜ ì¤‘ì‹¬ ë¦¬ë·°ë¡?ë¦¬ìŠ¤??ìµœì†Œ?
