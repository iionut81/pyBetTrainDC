param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$ApiToken = $env:API_FOOTBALL_TOKEN,
    [string]$Seasons = "2022,2023,2024,2025"
)

Set-Location $ProjectDir

if (-not $ApiToken -or $ApiToken.Trim().Length -eq 0) {
    throw "API_FOOTBALL_TOKEN is required for weekly corners refresh."
}

$historyCmd = @(
    "build_corners_history.py",
    "--api-key", $ApiToken,
    "--seasons", $Seasons,
    "--reserve", "2"
)
& $PythonExe @historyCmd
if ($LASTEXITCODE -ne 0) {
    throw "Corners history build failed with exit code $LASTEXITCODE"
}

$trainCmd = @(
    "train_corners_under_12_5.py",
    "--model", "nb",
    "--lookback-days", "365",
    "--retrain-days", "30"
)
& $PythonExe @trainCmd
if ($LASTEXITCODE -ne 0) {
    throw "Corners U12.5 training failed with exit code $LASTEXITCODE"
}

Write-Host "Weekly corners refresh + training completed."
