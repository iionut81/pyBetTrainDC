param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [int]$DaysAhead = 1,
    [double]$MinProb = 0.78,
    [double]$MinOdds = 1.10,
    [double]$MaxOdds = 1.35,
    [double]$MaxFairOdds = 1.30,
    [string]$Series = "1",
    [string]$ApiFootballToken = $env:API_FOOTBALL_TOKEN
)

Set-Location $ProjectDir

if (-not $ApiFootballToken -or $ApiFootballToken.Trim().Length -eq 0) {
    throw "API_FOOTBALL_TOKEN is required for daily corners predictions."
}

$targetDate = (Get-Date).Date.AddDays($DaysAhead).ToString("yyyy-MM-dd")

$cmd = @(
    "run_corners_daily.py",
    "--api-key", $ApiFootballToken,
    "--target-date", $targetDate,
    "--model", "nb",
    "--min-prob", "$MinProb",
    "--min-odds", "$MinOdds",
    "--max-odds", "$MaxOdds",
    "--max-fair-odds", "$MaxFairOdds",
    "--series", "$Series",
    "--insecure"
)

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Daily corners prediction run failed with exit code $LASTEXITCODE"
}

Write-Host "Daily corners U12.5 run completed for $targetDate (Series $Series)."
