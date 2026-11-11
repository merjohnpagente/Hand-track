#!/usr/bin/env bash
set -euo pipefail

# Build the static site used by both GitHub Pages and Vercel.
cd "$(dirname "$0")/.."

rm -rf dist
mkdir -p dist

cp web/index.html web/style.css web/app.js dist/
cp hand_landmarker.task selfie_segmenter.tflite dist/
touch dist/.nojekyll

echo "Built static site in dist/"
