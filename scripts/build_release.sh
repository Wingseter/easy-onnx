#!/usr/bin/env bash
# Builds the macOS arm64 release zip.
# Output: easyonnx-mac-arm64-v<VERSION>.zip in the repo root.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

VERSION=$(grep -E 'project\(easy-onnx VERSION' CMakeLists.txt \
            | sed -E 's/.*VERSION ([0-9.]+).*/\1/')

if [[ -z "${VERSION}" ]]; then
    echo "Failed to extract version from CMakeLists.txt" >&2
    exit 1
fi

ARCH="$(uname -m)"
if [[ "${ARCH}" != "arm64" ]]; then
    echo "WARN: Building on ${ARCH}, but the bundled ONNXRuntime dylib is arm64 only." >&2
fi

STAGE_NAME="easyonnx-mac-arm64"
STAGE_DIR="${REPO_ROOT}/stage/${STAGE_NAME}"
BUILD_DIR="${REPO_ROOT}/build-release"
ZIP_NAME="${STAGE_NAME}-v${VERSION}.zip"

echo "==> Cleaning ${BUILD_DIR} and ${STAGE_DIR}"
rm -rf "${BUILD_DIR}" "${STAGE_DIR}"

echo "==> Configuring (Release, arm64, install prefix=${STAGE_DIR})"
cmake -B "${BUILD_DIR}" -S "${REPO_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="${STAGE_DIR}"

echo "==> Building"
cmake --build "${BUILD_DIR}" --config Release -j

echo "==> Installing into staging directory"
cmake --install "${BUILD_DIR}"

echo "==> Zipping ${ZIP_NAME}"
rm -f "${REPO_ROOT}/${ZIP_NAME}"
# Use plain `zip` rather than `ditto --sequesterRsrc` so the archive doesn't
# carry __MACOSX resource-fork metadata. Preserves symlinks via `--symlinks`.
( cd "${REPO_ROOT}/stage" && \
  zip -r --symlinks -q "${REPO_ROOT}/${ZIP_NAME}" "${STAGE_NAME}" )

echo
echo "Done: ${REPO_ROOT}/${ZIP_NAME}"
ls -lh "${REPO_ROOT}/${ZIP_NAME}"
