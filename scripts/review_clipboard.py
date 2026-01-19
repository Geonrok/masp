"""GPT 검수용 클립보드 복사 스크립트."""
import subprocess
import sys
from pathlib import Path


def copy_to_clipboard(text: str) -> bool:
    """텍스트를 클립보드에 복사."""
    try:
        process = subprocess.Popen(
            ["clip"], stdin=subprocess.PIPE, shell=True
        )
        process.communicate(text.encode("utf-8"))
        return True
    except Exception as e:
        print(f"클립보드 복사 실패: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python review_clipboard.py <component_file> <test_file>")
        sys.exit(1)

    component_path = Path(sys.argv[1])
    test_path = Path(sys.argv[2])

    if not component_path.exists():
        print(f"파일 없음: {component_path}")
        sys.exit(1)
    if not test_path.exists():
        print(f"파일 없음: {test_path}")
        sys.exit(1)

    component_code = component_path.read_text(encoding="utf-8")
    test_code = test_path.read_text(encoding="utf-8")

    component_name = component_path.stem

    review_text = f"""## GPT 검수 요청: {component_name}.py

### 컴포넌트 개요
- **파일**: `{component_path.relative_to(component_path.parent.parent.parent.parent)}`
- **테스트**: `{test_path.relative_to(test_path.parent.parent.parent)}`

### 코드

```python
{component_code}
```

### 테스트

```python
{test_code}
```

### 검수 요청
위 코드와 테스트를 검수해주세요. 다음 형식으로 응답해주세요:

```
통과: true/false

필수보강:
1. (있으면 작성)

권장개선:
1. (있으면 작성)

코드품질:
  점수: X/10
  코멘트: "..."
```
"""

    if copy_to_clipboard(review_text):
        print("=" * 60)
        print("클립보드에 복사 완료. GPT-5.2에 붙여넣고 검수 결과를 알려주세요.")
        print("=" * 60)
    else:
        print("클립보드 복사 실패. 수동으로 복사하세요.")
        print(review_text)


if __name__ == "__main__":
    main()
