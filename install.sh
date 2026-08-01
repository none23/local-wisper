#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CLI_PATH="${PROJECT_ROOT}/wisper_cli.py"
PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"
TARGET_DIR="${HOME}/.local/bin"
TARGET_PATH="${TARGET_DIR}/lw"
CONFIG_DIR="${HOME}/.config/local-wisper"
ENV_PATH="${CONFIG_DIR}/env"
GLOSSARY_PATH="${CONFIG_DIR}/glossary.txt"

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
mkdir -p "${CONFIG_DIR}"
chmod 700 "${CONFIG_DIR}"

cat > "${TARGET_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON_PATH}" "${CLI_PATH}" "\$@"
EOF

chmod +x "${TARGET_PATH}"

if [[ ! -f "${ENV_PATH}" ]]; then
  cat > "${ENV_PATH}" <<EOF
# Optional OpenAI cleanup for local transcripts. Fill OPENAI_API_KEY and uncomment
# LW_POST_PROCESS_MODEL to enable post-processing.
export OPENAI_API_KEY=''
# export LW_POST_PROCESS_MODEL='gpt-5.4-nano'
export LW_POST_PROCESS_TIMEOUT='20'
export LW_POST_PROCESS_GLOSSARY_FILE='${GLOSSARY_PATH}'

# Sway wrapper defaults. These keep transcription local.
export LW_BACKEND='parakeet'
export LW_COMPUTE_TYPE='float16'
export LW_DEVICE='cuda'
export LW_VAD_FILTER='false'
export LW_OUTPUT_MODE='type'
EOF
  chmod 600 "${ENV_PATH}"
fi

if [[ ! -f "${GLOSSARY_PATH}" ]]; then
  cat > "${GLOSSARY_PATH}" <<'EOF'
[always]
dot env -> .env
package Jason -> package.json

[likely]
java script -> JavaScript
next jazz -> Next.js
next Jess -> Next.js
next JS -> Next.js
node jazz -> Node.js
node Jess -> Node.js
node JS -> Node.js
tail wind -> Tailwind
type script -> TypeScript

[contextual]

[terms]
OpenAI
Claude
Claude Code
Next.js
Node.js
TypeScript
JavaScript
React
TanStack Query
Tailwind CSS
PostgreSQL
Postgres
package.json
tsconfig.json
pnpm
Zod
Zustand
.env
EOF
  chmod 600 "${GLOSSARY_PATH}"
fi

echo "Installed: ${TARGET_PATH}"
echo "Config: ${ENV_PATH}"
echo "Glossary: ${GLOSSARY_PATH}"
if [[ ":${PATH}:" != *":${TARGET_DIR}:"* ]]; then
  echo "Note: ${TARGET_DIR} is not in PATH for this shell session."
  echo "Add this to your shell rc file:"
  echo "  export PATH=\"${TARGET_DIR}:\$PATH\""
fi

echo "Run: lw --help"
