#!/usr/bin/env bash
set -euo pipefail

lw_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lw_bin_dir="${HOME}/.local/bin"
lw_lib_dir="${HOME}/.local/lib/local-wisper"
lw_target="${lw_bin_dir}/lw"
lw_config_dir="${HOME}/.config/local-wisper"
lw_env_path="${lw_config_dir}/env"
lw_glossary_path="${lw_config_dir}/glossary.txt"
lw_cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}"
lw_package_cache="${lw_cache_base}/local-wisper/packages"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This experiment supports x86_64 Linux only." >&2
  exit 1
fi

for lw_command in cargo curl bsdtar pacman pacman-key; do
  if ! command -v "${lw_command}" >/dev/null 2>&1; then
    echo "Missing required command: ${lw_command}" >&2
    exit 1
  fi
done

echo "Building the release binary..."
cargo build --release --manifest-path "${lw_project_dir}/Cargo.toml"

mkdir -p "${lw_bin_dir}" "${lw_lib_dir}" "${lw_config_dir}" "${lw_package_cache}"
chmod 700 "${lw_config_dir}"

if [[ ! -f /usr/lib/libcudnn.so.9 && ! -f "${lw_lib_dir}/libcudnn.so.9" ]]; then
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

install -m755 "${lw_project_dir}/target/release/lw" "${lw_target}"

if [[ ! -f "${lw_env_path}" ]]; then
  {
    echo "export OPENAI_API_KEY=''"
    echo "export LW_POST_PROCESS_MODEL='gpt-5.6-luna'"
    echo "export LW_POST_PROCESS_TIMEOUT='20'"
    echo "export LW_POST_PROCESS_GLOSSARY_FILE='${lw_glossary_path}'"
    echo "export LW_BACKEND='parakeet'"
    echo "export LW_COMPUTE_TYPE='float16'"
    echo "export LW_DEVICE='cuda'"
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
echo "Run 'lw preload' to download the verified Parakeet model and load it on CUDA."
