#!/usr/bin/env sh
# AI-Review Hook Executor
# Usage: ai-review-hook.sh <MODE>
# MODE: autogen | verify

set -e

MODE="${1:-autogen}"
SCRIPT_PATH="C:/Users/고형석/Templates/ai-review-workflow/.ai-review/ai-review.ps1"

# Find pwsh executable
find_pwsh() {
    if [ -n "$PWSH_EXE" ] && [ -x "$PWSH_EXE" ]; then
        echo "$PWSH_EXE"
        return 0
    fi

    if command -v pwsh >/dev/null 2>&1; then
        command -v pwsh
        return 0
    fi

    # Hardcoded fallback paths
    for path in \
        "/usr/bin/pwsh" \
        "/usr/local/bin/pwsh" \
        "C:/Program Files/PowerShell/7/pwsh.exe" \
        "C:/Program Files (x86)/PowerShell/7/pwsh.exe"
    do
        if [ -x "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

# Main
PWSH=$(find_pwsh) || PWSH=""

if [ -z "$PWSH" ]; then
    if [ "$MODE" = "autogen" ]; then
        echo "[ai-review] WARNING: pwsh not found, skipping autogen"
        exit 0
    else
        echo "[ai-review] ERROR: pwsh not found, cannot run verify"
        exit 1
    fi
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    if [ "$MODE" = "autogen" ]; then
        echo "[ai-review] WARNING: ai-review.ps1 not found at $SCRIPT_PATH, skipping"
        exit 0
    else
        echo "[ai-review] ERROR: ai-review.ps1 not found at $SCRIPT_PATH"
        exit 1
    fi
fi

echo "[ai-review] Running $MODE mode..."
"$PWSH" -NoProfile -ExecutionPolicy Bypass -File "$SCRIPT_PATH" -Mode "$MODE"
