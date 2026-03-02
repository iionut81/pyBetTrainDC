param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python"
)

Set-Location $ProjectDir

$cmd = @(
    "import_transfermarkt.py",
    "--start-season-year", "2021",
    "--n-seasons", "5",
    "--output-csv", "simulations\seasons\transfermarkt_weekly_5s_snapshot.csv",
    "--merge-into", "data\historical\historical_matches_transfermarkt.csv",
    "--sleep-seconds", "0.3"
)

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Weekly league refresh failed with exit code $LASTEXITCODE"
}

# Optional API-Football enrichment for extra leagues (RO1, RS1, SA1).
# Set APIFOOTBALL_API_KEY env var before running task to enable this step.
if ($env:APIFOOTBALL_API_KEY) {
    $apiCmd = @(
        "import_api_football_history.py",
        "--api-key", $env:APIFOOTBALL_API_KEY,
        "--targets", "romania,serbia,saudi",
        "--n-seasons", "3",
        "--end-season", "2024",
        "--output-csv", "data\historical\historical_matches_api_football_extra.csv",
        "--merge-into", "data\historical\historical_matches_transfermarkt.csv",
        "--insecure"
    )
    & $PythonExe @apiCmd
    if ($LASTEXITCODE -ne 0) {
        throw "API-Football enrichment failed with exit code $LASTEXITCODE"
    }
    Write-Host "API-Football enrichment completed."
} else {
    Write-Host "APIFOOTBALL_API_KEY not set; skipping API-Football enrichment."
}

Write-Host "Weekly league refresh completed."
