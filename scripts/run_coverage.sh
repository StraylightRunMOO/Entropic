#!/usr/bin/env bash
# Build a coverage-instrumented tree, run tests, and fail if wfc.h line
# coverage is below 90%.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${COVERAGE_BUILD_DIR:-$ROOT/build-coverage}"
MIN_PCT="${COVERAGE_MIN_PCT:-90}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

LLVM_COV="$(command -v llvm-cov-22 || command -v llvm-cov || true)"
LLVM_PROFDATA="$(command -v llvm-profdata-22 || command -v llvm-profdata || true)"
if [[ -z "$LLVM_COV" || -z "$LLVM_PROFDATA" ]]; then
    echo "error: llvm-cov / llvm-profdata not found (need clang source-based coverage)" >&2
    exit 1
fi

NEON_FLAG=OFF
if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    NEON_FLAG=ON
fi

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="${CC:-clang}" \
    -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
    -DENTROPIC_ENABLE_COVERAGE=ON \
    -DENTROPIC_ENABLE_FUZZ=ON \
    -DENTROPIC_BUILD_TESTS=ON \
    -DENTROPIC_BUILD_PYTHON=ON \
    -DENTROPIC_USE_FAF=ON \
    -DENTROPIC_ENABLE_HIERARCHY=ON \
    -DENTROPIC_ENABLE_SUSPENDERS=ON \
    -DENTROPIC_ENABLE_NEON="$NEON_FLAG" \
    -DENTROPIC_ENABLE_AVX2=OFF \
    -DENABLE_ASAN=OFF \
    -DENABLE_UBSAN=OFF

cmake --build "$BUILD" -j "$JOBS"

PROFDIR="$BUILD/profraw"
rm -rf "$PROFDIR"
mkdir -p "$PROFDIR"

export LLVM_PROFILE_FILE="$PROFDIR/%p-%m.profraw"
(cd "$BUILD" && ctest --output-on-failure)

mapfile -t RAW < <(find "$PROFDIR" -name '*.profraw' 2>/dev/null || true)
if [[ ${#RAW[@]} -eq 0 ]]; then
    echo "error: no .profraw files produced" >&2
    exit 1
fi

"$LLVM_PROFDATA" merge -sparse "${RAW[@]}" -o "$BUILD/coverage.profdata"

BINS=()
for b in test_wfc_c test_wfc_cpp test_wfc_gtest test_quality test_hierarchy \
         test_suspenders fuzz_wfc_standalone test_gray_sobol; do
    if [[ -x "$BUILD/$b" ]]; then
        BINS+=("$BUILD/$b")
    fi
done
# Python extension, if present
shopt -s nullglob
for so in "$BUILD"/python/entropic/_entropic*.so "$BUILD"/_entropic*.so; do
    BINS+=("$so")
done
shopt -u nullglob

if [[ ${#BINS[@]} -eq 0 ]]; then
    echo "error: no instrumented binaries found" >&2
    exit 1
fi

OBJARGS=("${BINS[0]}")
for extra in "${BINS[@]:1}"; do
    OBJARGS+=(-object "$extra")
done

echo
echo "=== llvm-cov report (include/entropic) ==="
"$LLVM_COV" report "${OBJARGS[@]}" \
    -instr-profile="$BUILD/coverage.profdata" \
    -ignore-filename-regex='(_deps|/usr/|tests/|python/)' \
    "-sources=$ROOT/include/entropic" || true

REPORT="$("$LLVM_COV" report "${OBJARGS[@]}" \
    -instr-profile="$BUILD/coverage.profdata" \
    "-sources=$ROOT/include/entropic/wfc.h")"
echo "$REPORT"

# Parse the TOTAL line's line-coverage percent, or the wfc.h row.
PCT="$(echo "$REPORT" | awk '
    /wfc\.h/ { for (i=1;i<=NF;i++) if ($i ~ /%$/) { gsub("%","",$i); print $i; exit } }
')"
if [[ -z "$PCT" ]]; then
    PCT="$(echo "$REPORT" | awk '
        /TOTAL/ { for (i=1;i<=NF;i++) if ($i ~ /%$/) { gsub("%","",$i); print $i; exit } }
    ')"
fi

if [[ -z "$PCT" ]]; then
    echo "error: could not parse line coverage percent" >&2
    exit 1
fi

echo
echo "wfc.h line coverage: ${PCT}%  (minimum ${MIN_PCT}%)"
awk -v p="$PCT" -v m="$MIN_PCT" 'BEGIN { if (p+0 < m+0) exit 1; exit 0 }'
echo "Coverage target met."
