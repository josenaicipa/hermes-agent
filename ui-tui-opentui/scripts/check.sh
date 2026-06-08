#!/usr/bin/env bash
# Local CI for the native OpenTUI engine. Runs the dev-quality rails in order:
#   1. type-check  (tsc --noEmit)          — HARD gate
#   2. lint        (eslint, errors only)    — HARD gate (warnings are allowed)
#   3. demo.tsx    (FakeGateway headless)   — HARD gate (deterministic render)
#   4. demo.prompts.tsx (blocking prompts)  — HARD gate (Phase 4: the 4 gateway
#                    prompts + local confirm; request→render→answer→*.respond
#                    RPC and request→cancel→deny/cancel RPC for each).
#   5. demo.real.tsx (real Python gateway)  — transport gate; auto-skips when no
#                    Hermes python is resolvable, and passes on PASS|TRANSPORT OK
#                    (a full model reply needs API keys; transport-up is enough
#                    to catch regressions). Skip explicitly with
#                    HERMES_OPENTUI_SKIP_REAL=1.
#
# Usage:  bun run check   (or)   bash scripts/check.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
ROOT="$(pwd)"
FAIL=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$1"; }
bad()  { printf '\033[31m✗ %s\033[0m\n' "$1"; FAIL=1; }
skip() { printf '\033[33m• %s\033[0m\n' "$1"; }

step "1/5 type-check"
if bun run type-check; then ok "type-check clean"; else bad "type-check failed"; fi

step "2/5 lint (errors fail; warnings allowed)"
if bun run lint; then ok "lint clean (no errors)"; else bad "lint reported errors"; fi

step "3/5 demo.tsx (FakeGateway)"
if bun src/demo.tsx >/dev/null 2>&1 && grep -q 'leaked (\*\*): 0' demo-report.txt; then
  ok "FakeGateway transcript rendered, 0 markdown markers leaked"
else
  bad "FakeGateway demo failed"; [ -f demo-report.txt ] && sed 's/^/    /' demo-report.txt
fi

step "4/5 demo.prompts.tsx (blocking interactive prompts)"
if bun src/demo.prompts.tsx >/dev/null 2>&1 && grep -qE '^PASS' demo-prompts-report.txt; then
  ok "$(grep -E '^PASS' demo-prompts-report.txt | tail -1)"
else
  bad "prompts demo failed (deadlock-fix regression)"
  [ -f demo-prompts-report.txt ] && grep -E '✗|FAIL' demo-prompts-report.txt | sed 's/^/    /'
fi

step "5/5 demo.real.tsx (real Python gateway)"
if [ "${HERMES_OPENTUI_SKIP_REAL:-0}" = "1" ]; then
  skip "skipped (HERMES_OPENTUI_SKIP_REAL=1)"
else
  REAL_OUT="$(bun src/demo.real.tsx 2>&1)"
  REAL_RC=$?
  VERDICT="$(grep -E '^(PASS|TRANSPORT OK|BLOCKED|PARTIAL|FAIL)' demo-real-report.txt 2>/dev/null | tail -1)"
  if echo "$REAL_OUT" | grep -qiE 'Could not find the Python that runs Hermes'; then
    skip "skipped (no Hermes python resolvable on this box)"
  elif [ $REAL_RC -ne 0 ]; then
    bad "real demo exited non-zero ($REAL_RC)"
  elif echo "$VERDICT" | grep -qE '^(PASS|TRANSPORT OK)'; then
    ok "$VERDICT"
  else
    bad "transport regression: ${VERDICT:-no verdict}"
  fi
fi

step "result"
if [ $FAIL -eq 0 ]; then ok "all checks passed"; else bad "one or more checks failed"; fi
exit $FAIL
