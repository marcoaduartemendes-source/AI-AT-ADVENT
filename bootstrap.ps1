# Crypto Investment Bot — Windows one-shot installer
# Run in PowerShell. Handles Python, Git, clone, setup, and opens .env to edit.

$ErrorActionPreference = "Stop"
$repo = "https://github.com/marcoaduartemendes-source/AI-AT-ADVENT.git"
$branch = "claude/crypto-investment-app-TXp7Y"
$installDir = Join-Path $HOME "Desktop\AI-AT-ADVENT"

function Write-OK    { param($m) Write-Host "  [OK] $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "  [!!] $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "  [XX] $m" -ForegroundColor Red; exit 1 }
function Write-Step  { param($m) Write-Host ""; Write-Host "== $m ==" -ForegroundColor Cyan }

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Crypto Investment Bot - Windows Bootstrap" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# ── 1. Python ────────────────────────────────────────────────────────────────
Write-Step "Checking Python"
$pythonCmd = $null
foreach ($c in @("python", "py")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
        try {
            $v = & $c --version 2>&1
            if ($v -match "Python 3\.(\d+)") {
                $pythonCmd = $c
                Write-OK "$c -> $v"
                break
            }
        } catch { }
    }
}
if (-not $pythonCmd) {
    Write-Warn "Python not found. Installing via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
        $env:Path += ";$env:LOCALAPPDATA\Programs\Python\Python312;$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
        $pythonCmd = "python"
        Write-OK "Python installed. You may need to close and reopen PowerShell if commands fail."
    } else {
        Write-Fail "winget not available. Please install Python manually: https://www.python.org/downloads/  (check 'Add Python to PATH')"
    }
}

# ── 2. Git ───────────────────────────────────────────────────────────────────
Write-Step "Checking Git"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $gv = & git --version
    Write-OK "$gv"
} else {
    Write-Warn "Git not found. Installing via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install Git.Git -e --accept-source-agreements --accept-package-agreements
        $env:Path += ";$env:ProgramFiles\Git\cmd"
        Write-OK "Git installed."
    } else {
        Write-Fail "winget not available. Install Git manually: https://git-scm.com/download/win"
    }
}

# ── 3. Clone repo ────────────────────────────────────────────────────────────
Write-Step "Cloning project to $installDir"
if (Test-Path $installDir) {
    Write-Warn "Directory already exists. Pulling latest instead."
    Set-Location $installDir
    git fetch origin
    git checkout $branch
    git pull origin $branch
} else {
    git clone $repo $installDir
    Set-Location $installDir
    git checkout $branch
    Write-OK "Cloned branch $branch to $installDir"
}

# ── 4. Virtualenv + deps ─────────────────────────────────────────────────────
Write-Step "Installing Python dependencies (this takes ~1 min)"
if (-not (Test-Path "venv")) {
    & $pythonCmd -m venv venv
    Write-OK "Created virtualenv"
}
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
Write-OK "Dependencies installed"

# ── 5. Create .env ───────────────────────────────────────────────────────────
Write-Step "Setting up config file"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-OK "Created .env (copy of .env.example)"
} else {
    Write-Warn ".env already exists, not overwriting"
}
New-Item -ItemType Directory -Force -Path "data" | Out-Null
Write-OK "data\ directory ready"

# ── 6. Offline sanity check ──────────────────────────────────────────────────
Write-Step "Running offline sanity test (synthetic data, no keys)"
$env:SYNTHETIC_DATA = "true"
$env:SIMULATION = "true"
$env:MIN_CONFIDENCE = "0.3"
& ".\venv\Scripts\python.exe" src\main_trading.py --once 2>&1 | Select-String -Pattern "Mode|BUY|SELL|Cycle complete|CRYPTO TRADING" | Select-Object -First 15

# ── 7. Final instructions ────────────────────────────────────────────────────
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "  INSTALL COMPLETE" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The bot is installed at:" -ForegroundColor White
Write-Host "  $installDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening .env in Notepad now - paste your Coinbase key values on" -ForegroundColor White
Write-Host "lines 24 and 25, then File -> Save." -ForegroundColor White
Write-Host ""
Start-Sleep -Seconds 2
Start-Process notepad.exe -ArgumentList ".env"

Write-Host "After saving .env, come back to PowerShell and run:" -ForegroundColor White
Write-Host ""
Write-Host "  cd $installDir" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\activate" -ForegroundColor Yellow
Write-Host "  python scripts\check_setup.py" -ForegroundColor Yellow
Write-Host "  python src\main_trading.py --once" -ForegroundColor Yellow
Write-Host ""
Write-Host "To go live later: edit .env -> set DRY_RUN=false -> run again" -ForegroundColor White
Write-Host ""
