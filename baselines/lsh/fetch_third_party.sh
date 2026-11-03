#!/bin/bash
# Fetches the upstream sources the weighted-MinHash baseline builds on, at the exact
# commits recorded in the result CSVs (columns upstream_git, xxhash_git).
# Run from baselines/lsh/:   ./fetch_third_party.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party

if [[ ! -d third_party/dartminhash ]]; then
    git clone https://github.com/tobc/dartminhash third_party/dartminhash
fi
git -C third_party/dartminhash checkout ae0f2707df1e225810a673ec8f14958674eb6b97

if [[ ! -d third_party/xxhash ]]; then
    git clone https://github.com/Cyan4973/xxHash.git third_party/xxhash
fi
git -C third_party/xxhash checkout bbb27a5efb85b92a0486cf361a8635715a53f6ba   # v0.8.2

# libxxhash.a is needed by BagMinHash2. Override CC / CFLAGS if gcc-14 is not installed.
make -C third_party/xxhash lib CC="${CC:-gcc}" CFLAGS="${CFLAGS:--O3 -march=native}"
echo "third_party ready; now: make -C wmh CXX=${CXX:-g++-14}"
