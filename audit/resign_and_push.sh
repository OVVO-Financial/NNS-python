#!/usr/bin/env bash
# Re-sign the overnight branches with YOUR key and push them from your machine,
# so GitHub shows every commit as "Verified". Also pushes the R patch series to
# OVVO-Financial/NNS as a branch (you have push access there; the cloud session did not).
#
# Usage (Linux, macOS, or Git Bash on Windows):
#   bash resign_and_push.sh [work_dir]          # default work_dir: ~/nns-resign
#   RESET_AUTHOR=1 bash resign_and_push.sh      # also make YOU (your git config) the author
#   DRY_RUN=1 bash resign_and_push.sh           # re-sign locally, push nothing
#
# One-time setup for "Verified" with an SSH key (git >= 2.34):
#   git config --global gpg.format ssh
#   git config --global user.signingkey ~/.ssh/id_ed25519.pub
#   git config --global commit.gpgsign true
#   GitHub > Settings > SSH and GPG keys > New SSH key > Key type: "Signing Key" (same public key is fine)
#   git config --global user.email must be an email verified on your GitHub account.
#   The noreply address 99349713+gitRasheed@users.noreply.github.com always is; a private Gmail
#   address only works if "Block command line pushes that expose my email" is off.
set -euo pipefail

WORK="${1:-$HOME/nns-resign}"
BRANCH="claude/zen-cerf-jh8gs5"
R_BRANCH="fix/overnight-audit"
mkdir -p "$WORK"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git not found"
[ -n "$(git config user.email || true)" ] || die "set git config --global user.email first"
fmt="$(git config gpg.format || echo openpgp)"
[ -n "$(git config user.signingkey || true)" ] || die "set git config --global user.signingkey first (see header)"
say "Signing as $(git config user.name) <$(git config user.email)> with gpg.format=$fmt"

clone_or_update() {  # $1 repo name, $2 dir
  if [ -d "$2/.git" ]; then
    git -C "$2" fetch -q origin
  else
    git clone -q "git@github.com:OVVO-Financial/$1.git" "$2"
  fi
}

resign_branch() {  # $1 repo name, $2 base branch
  local name="$1" base_branch="$2" dir="$WORK/$1"
  say "$name: re-signing $BRANCH"
  clone_or_update "$name" "$dir"
  cd "$dir"
  git fetch -q origin "$BRANCH" "$base_branch"
  git checkout -q -B "$BRANCH" "origin/$BRANCH"
  local base
  base="$(git merge-base "origin/$base_branch" HEAD)"
  local amend="git commit -q --amend --no-edit --no-verify -S"
  [ "${RESET_AUTHOR:-0}" = "1" ] && amend="$amend --reset-author"
  local n
  n="$(git rev-list --count "$base"..HEAD)"
  echo "   $n commits since $(git rev-parse --short "$base") ($base_branch)"
  GIT_SEQUENCE_EDITOR=: git -c core.hooksPath=/dev/null rebase -q --exec "$amend" "$base"
  # every commit must now carry a signature
  local unsigned
  unsigned=0
  for c in $(git rev-list "$base"..HEAD); do
    git cat-file commit "$c" | grep -q '^gpgsig' || unsigned=$((unsigned + 1))
  done
  [ "$unsigned" = "0" ] || die "$name: $unsigned commits are unsigned"
  # content must be unchanged (only metadata was rewritten)
  git diff --quiet "origin/$BRANCH" HEAD || die "$name: tree changed during re-signing"
  echo "   all $n commits signed, tree identical to origin/$BRANCH"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "   DRY_RUN: not pushing"
  else
    git push --force-with-lease="$BRANCH:origin/$BRANCH" origin "$BRANCH"
  fi
  cd - >/dev/null
}

resign_branch NNS-python main
resign_branch NNS-core main

say "NNS (R): applying the patch series on NNS-Beta-Version as $R_BRANCH"
RDIR="$WORK/NNS"
clone_or_update NNS "$RDIR"
PATCHES="$WORK/NNS-python/audit/r-patches"
[ -d "$PATCHES/r-bugs" ] || die "patches not found at $PATCHES"
cd "$RDIR"
git fetch -q origin NNS-Beta-Version
git checkout -q -B "$R_BRANCH" origin/NNS-Beta-Version
echo "   base: $(git log -1 --format='%h %s')  (patches were made on 82a92eb 'NNS 13.3 Beta')"
# --keep-cr: the R sources use CRLF line endings; -S signs each applied commit
git am -q -S --keep-cr --whitespace=nowarn "$PATCHES"/r-bugs/*.patch
git am -q -S --keep-cr --whitespace=nowarn "$PATCHES"/cpp-core-bugs/*.patch
echo "   applied $(git rev-list --count origin/NNS-Beta-Version..HEAD) commits"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "   DRY_RUN: not pushing"
else
  git push -u origin "$R_BRANCH"
fi

say "Done. Branches: NNS-python + NNS-core $BRANCH (force-pushed, signed), NNS $R_BRANCH (new)."
echo "Open the PRs on GitHub when ready. If NNS-Beta-Version moved past 82a92eb and git am stops,"
echo "run 'git am --abort' and apply on 82a92eb instead: git checkout -B $R_BRANCH 82a92eb"
