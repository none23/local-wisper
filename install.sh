#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CLI_PATH="${PROJECT_ROOT}/wisper_cli.py"
PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"
TARGET_DIR="${HOME}/.local/bin"
TARGET_PATH="${TARGET_DIR}/lw"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer supports Linux only." >&2
  exit 1
fi

if [[ ! -f "${CLI_PATH}" ]]; then
  echo "Cannot find wisper_cli.py at ${CLI_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Missing virtualenv python at ${PYTHON_PATH}" >&2
  echo "Create it first:" >&2
  echo "  python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

cat > "${TARGET_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON_PATH}" "${CLI_PATH}" "\$@"
EOF

chmod +x "${TARGET_PATH}"

echo "Installed: ${TARGET_PATH}"
if [[ ":${PATH}:" != *":${TARGET_DIR}:"* ]]; then
  echo "Note: ${TARGET_DIR} is not in PATH for this shell session."
  echo "Add this to your shell rc file:"
  echo "  export PATH=\"${TARGET_DIR}:\$PATH\""
fi

echo "Run: lw --help"
