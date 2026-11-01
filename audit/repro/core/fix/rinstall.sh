#!/bin/bash
# Install the working tree of the R clone into ~/Rlib-patched from a clean copy
set -euo pipefail
SRC=/home/user/ovvo-financial/nns
STAGE=/home/user/audit/work/core/fix/rstage
rm -rf "$STAGE"; mkdir -p "$STAGE"
mkdir -p "$STAGE/NNS"; (cd "$SRC" && tar --exclude=.git --exclude="src/*.o" --exclude="src/*.so" -cf - .) | tar -xf - -C "$STAGE/NNS"
cd "$STAGE"
MAKEFLAGS=-j3 R CMD INSTALL --no-docs --no-html --no-multiarch -l ~/Rlib-patched NNS > "$STAGE/install.log" 2>&1 || { tail -40 "$STAGE/install.log"; exit 1; }
grep -c "g++" "$STAGE/install.log" | xargs echo "compiled units:"
tail -1 "$STAGE/install.log"
