#!/bin/sh
# Build the Liu reference interpreter (CPU backend). Linux and macOS.
# BLAS_FREE=1 builds against juzhen's handwritten kernels instead of
# OpenBLAS — dependency-free, ~5-15x slower on GEMM-heavy programs, and
# NOT bit-identical to the reference backend (goldens do not apply).
# It IS bit-identical across Linux machines (fixed accumulation order,
# fp-contract off; measured on two boxes whose OpenBLAS builds diverge:
# sha256-identical sica_instant output) — the cross-machine
# reproducibility mode, at the price of speed.
# The Juzhen backend lives in external/juzhen (git submodule, pinned):
# the pinned commit is part of the reproducibility statement —
# program text + seed + backend commit => bit-identical output.
# LOGGING_OFF drops Juzhen's spdlog dependency entirely (header + libs).
set -e
cd "$(dirname "$0")/.."
if [ ! -f external/juzhen/cpp/juzhen.hpp ]; then
  git submodule update --init external/juzhen   # deliberately not --recursive
fi
mkdir -p build_liu
if [ "$BLAS_FREE" = "1" ]; then
  # No external BLAS: juzhen's handwritten gemm/gemv (cpp/cpulinalg.hpp).
  # -ffp-contract=off is part of the mode — the kernels fix the accumulation
  # order, and fma contraction would change bits (aarch64 contracts by
  # default). NOT the reference backend: bits differ from the OpenBLAS
  # build, so goldens do not apply; quant checks are the test layer here.
  c++ -O2 -std=c++20 -w -DLOGGING_OFF -DJUZHEN_NO_BLAS -ffp-contract=off \
      -DPROJECT_DIR="\"$(pwd)\"" -I external/juzhen \
      interpreter/liu.cpp -o build_liu/liu
elif [ "$(uname)" = "Darwin" ]; then
  # brew install openblas
  OB="$(brew --prefix openblas)"
  c++ -O2 -std=c++20 -w -DLOGGING_OFF -DPROJECT_DIR="\"$(pwd)\"" \
      -I external/juzhen -I"$OB/include" \
      interpreter/liu.cpp -o build_liu/liu \
      -L"$OB/lib" -lopenblas
else
  # sudo apt install g++ libopenblas-dev
  g++ -O2 -std=c++20 -w -DLOGGING_OFF -DPROJECT_DIR="\"$(pwd)\"" \
      -I external/juzhen \
      interpreter/liu.cpp -o build_liu/liu \
      -lopenblas
fi
echo "built: build_liu/liu"
