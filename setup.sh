#!/usr/bin/env bash
# One-shot setup for Mac/Linux.
# Run from the project root: bash setup.sh
set -e

GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; NC='\033[0m'
ok()    { printf "  ${GREEN}✓${NC} %s\n" "$1"; }
warn()  { printf "  ${YELLOW}!${NC} %s\n" "$1"; }
fail()  { printf "  ${RED}✗${NC} %s\n" "$1"; exit 1; }

echo ""
echo "═══ Crypto Investment Bot — Local Setup ═══"
echo ""

# ── 1. Check Python ──────────────────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    fail "Python not found. Install it from https://www.python.org/downloads/ then re-run this script."
fi
PYV=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "Python $PYV detected"

# ── 2. Virtual environment ───────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    $PY -m venv venv
    ok "Created virtualenv ./venv"
else
    ok "Virtualenv ./venv already exists"
fi
# shellcheck disable=SC1091
source venv/bin/activate
ok "Activated virtualenv"

# ── 3. Install requirements ──────────────────────────────────────────────────
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
ok "Installed dependencies from requirements.txt"

# ── 4. Create .env from template if missing ──────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    chmod 600 .env
    ok "Created .env (from .env.example, mode 600)"
else
    warn ".env already exists — not overwriting"
fi

# ── 5. Make data dir ─────────────────────────────────────────────────────────
mkdir -p data
ok "Data directory ready (./data)"

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "  NEXT STEPS"
echo "═════════════════════════════════════════════════════════════"
echo ""
echo "  1.  Edit .env and paste your Coinbase API key values:"
echo "      (Mac)     open -e .env"
echo "      (Linux)   nano .env"
echo ""
echo "      → Update lines 24 and 25 with your real key values."
echo ""
echo "  2.  Validate everything:"
echo "      source venv/bin/activate"
echo "      python scripts/check_setup.py"
echo ""
echo "  3.  Run one paper-trading cycle:"
echo "      python src/main_trading.py --once"
echo ""
echo "  4.  Run continuously (still paper trading):"
echo "      python src/main_trading.py"
echo ""
echo "  5.  Go live (only when ready!):"
echo "      Edit .env → set DRY_RUN=false"
echo "      python src/main_trading.py"
echo ""
echo "═════════════════════════════════════════════════════════════"
echo ""
