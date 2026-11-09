#!/usr/bin/env bash
# Build the predictive-agent Nix image and load it into the local docker daemon.
#
# Requires:
#   - Nix with flakes enabled (https://nixos.org/download.html)
#   - a running docker daemon. On macOS use colima:
#       brew install colima && colima start
#       # provides ~/.colima/default/docker.sock
#
# Usage:
#   ./scripts/build-nix-image.sh
#
# The image is tagged predictive-agent:v4.0.8-nix and loaded into `docker images`.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="v4.0.8.1-nix"
OUT="result-image"

rm -rf "$OUT"
echo "[build] nix build .#predictive-agent -o $OUT"
nix build .#predictive-agent -o "$OUT"

# dockerTools.buildLayeredImage may emit either a tarball or a directory
# containing the image tar — handle both.
if [ -d "$OUT" ]; then
  TAR=$(find "$OUT" -maxdepth 1 \( -name '*.tar' -o -name '*.tar.gz' \) | head -1)
  if [ -z "$TAR" ]; then
    echo "[error] no image tar found inside $OUT" >&2
    exit 1
  fi
else
  TAR="$OUT"
fi

echo "[build] docker load -i $TAR"
docker load -i "$TAR"

echo "[ok] loaded predictive-agent:${TAG}"
echo "Run with: docker run --rm -v /var/run/docker.sock:/var/run/docker.sock predictive-agent:${TAG}"
echo "   or:    docker compose -f docker-compose.yml up"
