#!/usr/bin/env bash
# Fetch and patch TrackEval, the official MOTChallenge metric implementation.
#
# TrackEval is not vendored into this repository (it is MIT-licensed but
# large, and pinning a clone is clearer than copying its source). It is
# also unmaintained against modern NumPy: it still uses np.float, np.int
# and np.bool, which NumPy removed in 2.0. Those were plain aliases for
# the Python builtins, so replacing them changes no behaviour — but
# without the replacement every evaluation dies with
#   AttributeError: module 'numpy' has no attribute 'float'
#
# Run once. Safe to re-run: it re-clones from scratch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO_ROOT/external/TrackEval"
UPSTREAM="https://github.com/JonathonLuiten/TrackEval.git"

# Pinned so the metric implementation cannot change under the numbers in
# docs/. Bump deliberately, and re-run the benchmark when you do.
COMMIT="12c8791b303e0a0b50f753af204249e622d0281a"

if [ -d "$TARGET" ]; then
  echo "Removing existing $TARGET"
  rm -rf "$TARGET"
fi

mkdir -p "$REPO_ROOT/external"
echo "Cloning TrackEval…"
git clone --quiet "$UPSTREAM" "$TARGET"
git -C "$TARGET" checkout --quiet "$COMMIT"
echo "Pinned at $(git -C "$TARGET" rev-parse --short HEAD)"

echo "Patching removed NumPy aliases…"
BEFORE=$( (grep -rn --include='*.py' -E 'np\.(float|int|bool|object|str)\b' "$TARGET" || true) | wc -l | tr -d ' ')
find "$TARGET" -name '*.py' -print0 \
  | xargs -0 perl -pi -e 's/\bnp\.(float|int|bool|object|str)\b/$1/g'
AFTER=$( (grep -rn --include='*.py' -E 'np\.(float|int|bool|object|str)\b' "$TARGET" || true) | wc -l | tr -d ' ')

echo "Replaced $BEFORE alias use(s); $AFTER remaining."
if [ "$AFTER" -ne 0 ]; then
  echo "FAIL  patch left $AFTER unpatched alias(es)" >&2
  exit 1
fi

echo "Checking it imports…"
"$REPO_ROOT/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$TARGET')
import trackeval
trackeval.metrics.HOTA({'METRICS': ['HOTA'], 'THRESHOLD': 0.5})
print('TrackEval ready.')
"
