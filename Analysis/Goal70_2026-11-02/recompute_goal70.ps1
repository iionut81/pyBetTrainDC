param([string]$ReportDirectory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$rows = @(Import-Csv -LiteralPath (Join-Path $ReportDirectory 'goal_minutes_verified.csv'))
$expected = Get-Content -LiteralPath (Join-Path $ReportDirectory 'calculations.json') -Raw -Encoding UTF8 | ConvertFrom-Json

function Get-Minutes([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return }
    foreach ($part in ($Value -split ';')) {
        if ($part -notmatch '^(\d{1,2})(\+\d{1,2})?$') { throw "Invalid goal minute: $part" }
        $base = [int](($part -split '\+')[0])
        if ($base -gt 90) { throw "Extra-time goal in regular-time dataset: $part" }
        $part
    }
}
function Count-By70([string]$Value) {
    @((Get-Minutes $Value) | Where-Object { [int](($_ -split '\+')[0]) -le 70 }).Count
}

$seen = @{}
foreach ($row in $rows) {
    $key = "$($row.date)|$($row.league)|$($row.home)|$($row.away)"
    if ($seen.ContainsKey($key)) { throw "Duplicate match: $key" }
    $seen[$key] = $true
    if ($row.date -ge '2026-11-02') { throw "Future match in historical input: $key" }
    $hMinutes = @(Get-Minutes $row.home_goal_minutes)
    $aMinutes = @(Get-Minutes $row.away_goal_minutes)
    if ($hMinutes.Count -ne [int]$row.home_score -or $aMinutes.Count -ne [int]$row.away_score) {
        throw "Goal list does not reconcile with score: $key"
    }
    $h70 = Count-By70 $row.home_goal_minutes
    $a70 = Count-By70 $row.away_goal_minutes
    if ($h70 -ne [int]$row.home_goals_by_70 -or $a70 -ne [int]$row.away_goals_by_70) {
        throw "By-70 count mismatch: $key"
    }
    if (($h70 + $a70 -gt 0) -ne ($row.goal_by_70 -eq 'true')) { throw "Outcome mismatch: $key" }
    $first = @($hMinutes + $aMinutes | Sort-Object {
        $parts = $_ -split '\+'
        [double]$parts[0] + $(if ($parts.Count -eq 2) { [double]$parts[1] / 100 } else { 0 })
    } | Select-Object -First 1)
    $firstText = if ($first.Count) { [string]$first[0] } else { '' }
    if ($firstText -ne $row.first_goal) { throw "First-goal mismatch: $key" }
}
if ((Count-By70 '45+2;70;71;90+1') -ne 2) { throw 'Boundary check failed' }
if ($rows.Count -ne [int]$expected.unique_matches) { throw 'Dataset row count mismatch' }

function Get-TeamStats([string]$Team, [string]$League, [string]$Venue = 'all') {
    $sample = @($rows | Where-Object {
        $_.league -eq $League -and $_.season -eq '2026/27' -and
        (($Venue -eq 'home' -and $_.home -eq $Team) -or
         ($Venue -eq 'away' -and $_.away -eq $Team) -or
         ($Venue -eq 'all' -and ($_.home -eq $Team -or $_.away -eq $Team)))
    })
    $gf = 0; $ga = 0
    foreach ($match in $sample) {
        if ($match.home -eq $Team) {
            $gf += Count-By70 $match.home_goal_minutes
            $ga += Count-By70 $match.away_goal_minutes
        } else {
            $gf += Count-By70 $match.away_goal_minutes
            $ga += Count-By70 $match.home_goal_minutes
        }
    }
    [pscustomobject]@{ n = $sample.Count; gf70 = $gf; ga70 = $ga }
}
function Get-Lambda($A, $B) {
    if ($A.n -lt 1 -or $B.n -lt 1) { throw 'Empty model sample' }
    ($A.gf70 / $A.n + $B.ga70 / $B.n + $B.gf70 / $B.n + $A.ga70 / $A.n) / 2
}
$computed = foreach ($match in $expected.results) {
    $a = Get-TeamStats $match.home $match.league
    $b = Get-TeamStats $match.away $match.league
    $aVenue = Get-TeamStats $match.home $match.league 'home'
    $bVenue = Get-TeamStats $match.away $match.league 'away'
    $ls = Get-Lambda $a $b
    $lv = Get-Lambda $aVenue $bVenue
    $lc = ($ls + $lv) / 2
    $p = 1 - [math]::Exp(-$lc)
    $stress = 1 - [math]::Exp(-0.8 * $lc)
    if ([math]::Abs($p - $match.p_raw) -gt 1e-12 -or
        [math]::Abs($stress - $match.p_stress) -gt 1e-12 -or
        [math]::Abs(1 / $p - $match.fair_odds) -gt 1e-12) { throw 'Model reproduction mismatch' }
    [pscustomobject]@{ home = $match.home; away = $match.away; lambda_season = $ls;
        lambda_venue = $lv; lambda_combined = $lc; p_raw = $p; p_stress = $stress; fair_odds = 1 / $p }
}
$computed | ConvertTo-Json -Depth 5
"PASS: $($rows.Count) unique historical scores reconciled; minute boundaries, outcomes and three models reproduced."
