#!/usr/bin/env bash
# Tear down a git worktree created by the Zed editor (see worktree-symlinks.sh):
#
#   1. remove the worktree checkout        pfund.ai/worktrees/pfund/<name>/pfund
#   2. delete the local branch              <name>
#   3. delete the remote branch             origin/<name>
#   4. remove the worktree folder           pfund.ai/worktrees/pfund/<name>
#
# Step 4 only removes symlinks and empty directories. If anything else is in
# the folder it is left in place with a warning.
#
# Usage: scripts/worktree-remove.sh <name>
#        pixi run rmtree <name>

set -e

NAME=${1%/}
[ -n "$NAME" ] || { echo "usage: $0 <name>"; exit 1; }
case "$NAME" in main|master|*..*|/*) echo "refusing to remove '$NAME'"; exit 1 ;; esac

COMMON=$(git rev-parse --path-format=absolute --git-common-dir)
MAIN_CHECKOUT=$(dirname "$COMMON")            # e.g. pfund.ai/pfund
MAIN=$(dirname "$MAIN_CHECKOUT")              # e.g. pfund.ai
REPO_NAME=$(basename "$MAIN_CHECKOUT")        # e.g. pfund
WORKTREES_ROOT="$MAIN/worktrees/$REPO_NAME"   # e.g. pfund.ai/worktrees/pfund
DIR="$WORKTREES_ROOT/$NAME"
WORKTREE="$DIR/$REPO_NAME"

# 1. worktree
if [ -d "$WORKTREE" ]; then
  git worktree remove --force "$WORKTREE"
  echo "worktree: removed $WORKTREE"
else
  echo "worktree: none at $WORKTREE"
fi
git worktree prune

# 2. local branch
if git show-ref --verify --quiet "refs/heads/$NAME"; then
  git branch -D "$NAME"
  echo "branch:   deleted local '$NAME'"
else
  echo "branch:   no local '$NAME'"
fi

# 3. remote branch
if git ls-remote --exit-code --heads origin "$NAME" >/dev/null 2>&1; then
  git push origin --delete "$NAME"
  echo "remote:   deleted origin/$NAME"
else
  echo "remote:   no origin/$NAME"
fi

# 4. folder (symlinks and empty dirs only)
if [ -d "$DIR" ]; then
  find "$DIR" -type l -delete
  find "$DIR" -name .DS_Store -type f -delete
  d="$DIR"
  while [ "$d" != "$WORKTREES_ROOT" ] && rmdir "$d" 2>/dev/null; do
    echo "folder:   removed $d"
    d=$(dirname "$d")
  done
  if [ -e "$DIR" ]; then echo "folder:   left $DIR alone (contains other files)"; fi
else
  echo "folder:   none at $DIR"
fi
