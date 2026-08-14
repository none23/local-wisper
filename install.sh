#!/usr/bin/env bash
set -euo pipefail

lw_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lw_bin_dir="${HOME}/.local/bin"
lw_lib_dir="${HOME}/.local/lib/local-wisper"
lw_target="${lw_bin_dir}/lw"
lw_ort_shared="libonnxruntime_providers_shared.so"
lw_ort_cuda="libonnxruntime_providers_cuda.so"
lw_config_dir="${HOME}/.config/local-wisper"
lw_env_path="${lw_config_dir}/env"
lw_glossary_path="${lw_config_dir}/glossary.txt"
lw_cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}"
lw_package_cache="${lw_cache_base}/local-wisper/packages"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This experiment supports x86_64 Linux only." >&2
  exit 1
fi

for lw_command in cargo curl readlink; do
  if ! command -v "${lw_command}" >/dev/null 2>&1; then
    echo "Missing required command: ${lw_command}" >&2
    exit 1
  fi
done

echo "Building the release binary..."
cargo build --release --locked --manifest-path "${lw_project_dir}/Cargo.toml"

for lw_ort_library in "${lw_ort_shared}" "${lw_ort_cuda}"; do
  if [[ ! -f "${lw_project_dir}/target/release/${lw_ort_library}" ]]; then
    echo "Release build did not produce ${lw_ort_library}." >&2
    exit 1
  fi
done

mkdir -p "${lw_bin_dir}" "${lw_lib_dir}" "${lw_config_dir}" "${lw_package_cache}"
chmod 700 "${lw_config_dir}"

lw_has_nvidia=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  lw_has_nvidia=true
fi

if [[ "${lw_has_nvidia}" == true && ! -f /usr/lib/libcudnn.so.9 && ! -f "${lw_lib_dir}/libcudnn.so.9" ]]; then
  for lw_command in bsdtar pacman pacman-key; do
    if ! command -v "${lw_command}" >/dev/null 2>&1; then
      echo "CUDA is available, but cuDNN 9 is missing and ${lw_command} cannot install it." >&2
      echo "Install cuDNN 9 for this system, then rerun install.sh." >&2
      exit 1
    fi
  done
  echo "Downloading the signed Manjaro cuDNN package..."
  lw_cudnn_url="$(pacman -Sp --print-format '%l' cudnn | tail -n 1)"
  if [[ -z "${lw_cudnn_url}" ]]; then
    echo "Could not resolve the cudnn package from the configured pacman repositories." >&2
    exit 1
  fi
  lw_cudnn_package="${lw_package_cache}/${lw_cudnn_url##*/}"
  curl --fail --location --continue-at - --output "${lw_cudnn_package}" "${lw_cudnn_url}"
  curl --fail --location --output "${lw_cudnn_package}.sig" "${lw_cudnn_url}.sig"
  pacman-key --verify "${lw_cudnn_package}.sig" "${lw_cudnn_package}"

  lw_extract_dir="$(mktemp -d)"
  trap 'rm -rf -- "${lw_extract_dir}"' EXIT
  bsdtar -xf "${lw_cudnn_package}" -C "${lw_extract_dir}"
  cp -a "${lw_extract_dir}"/usr/lib/libcudnn*.so.9* "${lw_lib_dir}/"
  rm -rf -- "${lw_extract_dir}"
  trap - EXIT
fi

while read -r lw_legacy_pid; do
  [[ -n "${lw_legacy_pid}" ]] || continue
  lw_legacy_command="$(tr '\0' ' ' <"/proc/${lw_legacy_pid}/cmdline" 2>/dev/null || true)"
  if [[ "${lw_legacy_command}" == *"/local-wisper/"*"transcribe_daemon.py"* ]]; then
    echo "Stopping legacy Python model process ${lw_legacy_pid}..."
    kill "${lw_legacy_pid}" 2>/dev/null || true
  fi
done < <(pgrep -u "$(id -u)" -f 'transcribe_daemon\.py' || true)

while read -r lw_native_pid; do
  [[ -n "${lw_native_pid}" ]] || continue
  lw_native_exe="$(readlink -f "/proc/${lw_native_pid}/exe" 2>/dev/null || true)"
  if [[ "${lw_native_exe}" == "${lw_target}" ]]; then
    echo "Stopping installed native model process ${lw_native_pid}..."
    kill "${lw_native_pid}" 2>/dev/null || true
    for _ in {1..50}; do
      [[ ! -e "/proc/${lw_native_pid}" ]] && break
      sleep 0.1
    done
    if [[ -e "/proc/${lw_native_pid}" ]]; then
      kill -KILL "${lw_native_pid}" 2>/dev/null || true
    fi
  fi
done < <(pgrep -u "$(id -u)" -f '(^|/)lw __daemon( |$)' || true)

install -m755 "${lw_project_dir}/target/release/lw" "${lw_target}"
for lw_ort_library in "${lw_ort_shared}" "${lw_ort_cuda}"; do
  install -m755 \
    -T "$(readlink -f "${lw_project_dir}/target/release/${lw_ort_library}")" \
    "${lw_lib_dir}/${lw_ort_library}"
  ln -sfn \
    "../lib/local-wisper/${lw_ort_library}" \
    "${lw_bin_dir}/${lw_ort_library}"
done

if [[ ! -f "${lw_env_path}" ]]; then
  {
    echo "export OPENAI_API_KEY=''"
    echo "export LW_POST_PROCESS_MODEL=''"
    echo "export LW_POST_PROCESS_TIMEOUT='20'"
    echo "export LW_POST_PROCESS_GLOSSARY_FILE='${lw_glossary_path}'"
    echo "export LW_BACKEND='parakeet'"
    echo "export LW_DEVICE='auto'"
    echo "export LW_VAD_FILTER='false'"
    echo "export LW_OUTPUT_MODE='type'"
  } >"${lw_env_path}"
  chmod 600 "${lw_env_path}"
fi

if [[ ! -f "${lw_glossary_path}" ]]; then
  install -m600 "${lw_project_dir}/glossary.example.txt" "${lw_glossary_path}"
fi

echo "Caching the BAML 0.16 runtime..."
"${lw_target}" sway-cancel

echo "Installed ${lw_target}"
echo "Run 'lw preload' to select the best available runtime and load Parakeet."
