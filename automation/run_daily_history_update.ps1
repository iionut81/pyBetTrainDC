param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python"
)

Set-Location $ProjectDir

$cmd = @(
    "import_transfermarkt.py",
    "--start-season-year", "2025",
    "--n-seasons", "1",
    "--output-csv", "simulations\transfermarkt_daily_2025_26_snapshot.csv",
    "--merge-into", "data\historical\historical_matches_transfermarkt.csv",
    "--sleep-seconds", "0.3"
)

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Daily history update failed with exit code $LASTEXITCODE"
}

Write-Host "History update completed."
