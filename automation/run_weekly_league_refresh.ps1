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

Write-Host "Weekly league refresh completed."
