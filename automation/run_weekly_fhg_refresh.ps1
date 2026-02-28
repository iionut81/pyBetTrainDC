param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$ApiToken = $env:API_FOOTBALL_TOKEN
)

Set-Location $ProjectDir

if (-not $ApiToken -or $ApiToken.Trim().Length -eq 0) {
    throw "API_FOOTBALL_TOKEN is required for weekly FHG refresh."
}

$cmd = @(
    "build_fhg_history.py",
    "--api-key", $ApiToken,
    "--start-season", "2022",
    "--end-season", "2024",
    "--requests-per-minute", "8",
    "--max-retries", "2",
    "--retry-sleep-seconds", "65",
    "--out-history", "simulations\FHG\data\fhg_history.csv",
    "--out-ratios", "simulations\FHG\data\fhg_league_ratios.csv",
    "--insecure"
)

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "FHG weekly history build failed with exit code $LASTEXITCODE"
}

# Add current-season (2025/26) rows from Flashscore until API season access is available.
$flashCmd = @(
    "import_flashscore_fhg_current.py",
    "--season-start", "2025",
    "--out-csv", "simulations\FHG\data\fhg_history_2025_26_flashscore.csv",
    "--merge-into", "simulations\FHG\data\fhg_history.csv",
    "--insecure"
)
& $PythonExe @flashCmd
if ($LASTEXITCODE -ne 0) {
    throw "FHG Flashscore current-season import failed with exit code $LASTEXITCODE"
}

# Rebuild ratios from merged history (API 2022-2024 + Flashscore 2025/26).
$ratioCmd = @(
    "rebuild_fhg_ratios.py",
    "--history-csv", "simulations\FHG\data\fhg_history.csv",
    "--out-ratios", "simulations\FHG\data\fhg_league_ratios.csv"
)
& $PythonExe @ratioCmd
if ($LASTEXITCODE -ne 0) {
    throw "FHG ratio rebuild failed with exit code $LASTEXITCODE"
}

$calCmd = @(
    "train_fhg_calibration.py",
    "--history-csv", "simulations\FHG\data\fhg_history.csv",
    "--ratios-csv", "simulations\FHG\data\fhg_league_ratios.csv",
    "--out-csv", "simulations\FHG\data\fhg_calibration.csv"
)
& $PythonExe @calCmd
if ($LASTEXITCODE -ne 0) {
    throw "FHG calibration training failed with exit code $LASTEXITCODE"
}

$biasCmd = @(
    "train_fhg_league_bias.py",
    "--history-csv", "simulations\FHG\data\fhg_history.csv",
    "--ratios-csv", "simulations\FHG\data\fhg_league_ratios.csv",
    "--calibration-csv", "simulations\FHG\data\fhg_calibration.csv",
    "--out-csv", "simulations\FHG\data\fhg_league_bias.csv"
)
& $PythonExe @biasCmd
if ($LASTEXITCODE -ne 0) {
    throw "FHG league bias training failed with exit code $LASTEXITCODE"
}

$auditCmd = @(
    "audit_fhg.py",
    "--history-csv", "simulations\FHG\data\fhg_history.csv",
    "--ratios-csv", "simulations\FHG\data\fhg_league_ratios.csv"
)
& $PythonExe @auditCmd
if ($LASTEXITCODE -ne 0) {
    throw "FHG weekly audit failed with exit code $LASTEXITCODE"
}

Write-Host "FHG weekly refresh + audit completed."
