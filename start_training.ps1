# ==============================================
#  FL TRAINING LAUNCHER - start_training.ps1
#  Starts server + 4 clients in separate windows
# ==============================================
#
# Usage: Right-click → "Run with PowerShell"
#        OR from terminal: .\start_training.ps1
#
# ==============================================

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "$ProjectDir\venv\Scripts\python.exe"

# --- Validate venv exists ---
if (-not (Test-Path $Python)) {
    Write-Host "[ERROR] venv not found at: $Python" -ForegroundColor Red
    Write-Host "Run: python -m venv venv && .\venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   FEDERATED LEARNING TRAINING LAUNCHER    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Python : $Python" -ForegroundColor Gray
Write-Host "  Dir    : $ProjectDir" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Start Server ---
Write-Host "[1/5] Starting FL Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir'; Write-Host 'FL SERVER' -ForegroundColor Cyan; & '$Python' run_server.py"
) -WindowStyle Normal

Write-Host "      Server window opened." -ForegroundColor Gray
Write-Host "      Waiting 5 seconds for server to start..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# --- Start Client 1 ---
Write-Host "[2/5] Starting Client 1..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir'; Write-Host 'CLIENT 1' -ForegroundColor Yellow; & '$Python' run_client.py --client_id client_1"
) -WindowStyle Normal
Start-Sleep -Seconds 2

# --- Start Client 2 ---
Write-Host "[3/5] Starting Client 2..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir'; Write-Host 'CLIENT 2' -ForegroundColor Yellow; & '$Python' run_client.py --client_id client_2"
) -WindowStyle Normal
Start-Sleep -Seconds 2

# --- Start Client 3 ---
Write-Host "[4/5] Starting Client 3..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir'; Write-Host 'CLIENT 3' -ForegroundColor Yellow; & '$Python' run_client.py --client_id client_3"
) -WindowStyle Normal
Start-Sleep -Seconds 2

# --- Start Client 4 ---
Write-Host "[5/5] Starting Client 4..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectDir'; Write-Host 'CLIENT 4' -ForegroundColor Yellow; & '$Python' run_client.py --client_id client_4"
) -WindowStyle Normal

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  All 5 windows launched!" -ForegroundColor Green
Write-Host "  5 rounds of FedProx training will run." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Enter to close this launcher..." -ForegroundColor Gray
Read-Host
