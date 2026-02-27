param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$RatingsPkl = "data\historical\team_ratings.pkl",
    [int]$DaysAhead = 1,
    [double]$MinOdds = 1.10,
    [double]$MaxOdds = 1.25,
    [string]$Series = "1",
    [string]$ApiToken = $env:FOOTBALL_DATA_TOKEN,
    [string]$ApiFootballToken = $env:API_FOOTBALL_TOKEN
)

Set-Location $ProjectDir

$targetDate = (Get-Date).Date.AddDays($DaysAhead).ToString("yyyy-MM-dd")
$apiUrl = ""
$activeApiToken = ""

if ($ApiFootballToken -and $ApiFootballToken.Trim().Length -gt 0) {
    $apiUrl = "https://v3.football.api-sports.io/fixtures?date=$targetDate"
    $activeApiToken = $ApiFootballToken
}
elseif ($ApiToken -and $ApiToken.Trim().Length -gt 0) {
    $apiUrl = "https://api.football-data.org/v4/matches?dateFrom=$targetDate&dateTo=$targetDate"
    $activeApiToken = $ApiToken
}
$evalOut = "simulations\evaluations\$Series.1_Today_Evaluations.csv"
$recOut = "simulations\recommendations\$Series.2_Today_Recommendations.csv"

New-Item -ItemType Directory -Force -Path "simulations\evaluations" | Out-Null
New-Item -ItemType Directory -Force -Path "simulations\recommendations" | Out-Null

$cmd = @(
    "main.py",
    "--provider", "auto",
    "--target-date", $targetDate,
    "--ratings-pkl", $RatingsPkl,
    "--output-csv", $recOut,
    "--all-matches-csv", $evalOut,
    "--min-odds", "$MinOdds",
    "--max-odds", "$MaxOdds",
    "--insecure"
)

if ($activeApiToken -and $activeApiToken.Trim().Length -gt 0) {
    $cmd += @("--fixtures-api-url", $apiUrl, "--api-key", $activeApiToken)
}

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Daily prediction run failed with exit code $LASTEXITCODE"
}

Write-Host "Prediction run completed for $targetDate (Series $Series)."
