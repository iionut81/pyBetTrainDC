param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$HistoryCsv = "data\historical\historical_matches_transfermarkt.csv",
    [string]$OutputPkl = "data\historical\team_ratings.pkl"
)

Set-Location $ProjectDir

$cmd = @(
    "train_team_ratings.py",
    "--history-csv", $HistoryCsv,
    "--output-pkl", $OutputPkl,
    "--lookback-days", "270",
    "--decay-xi", "0.0025"
)

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Weekly retrain failed with exit code $LASTEXITCODE"
}

Write-Host "Weekly ratings retrain completed."
