param([string]$AsOf = '2026-11-01')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$folder = Join-Path $root "Analysis/Football70_Corners1H_Audit_$AsOf"
$summary = Get-Content -LiteralPath (Join-Path $folder 'summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = @(Import-Csv -LiteralPath (Join-Path $folder 'source_manifest.csv'))
foreach ($m in $manifest) {
    $actualHash = (Get-FileHash -LiteralPath (Join-Path $root $m.path) -Algorithm SHA256).Hash
    if ($actualHash -ne $m.sha256) { throw "Input changed since audit: $($m.path)" }
}
$pred = @(Import-Csv -LiteralPath (Join-Path $folder 'corners_1h_brentford_walkforward_predictions.csv'))
$top = @(Import-Csv -LiteralPath (Join-Path $folder 'corners_1h_daily_top3_retrospective.csv'))
if ($pred.Count -ne $summary.corners_backtest_predictions) { throw 'Prediction count mismatch.' }
$bad = @($pred | Where-Object {
    $_.match_date -ge $AsOf -or [int]$_.n_home_season -lt $summary.min_prior_season_matches -or
    [int]$_.n_away_season -lt $summary.min_prior_season_matches -or
    [int]$_.n_home_venue -lt $summary.min_prior_venue_matches -or
    [int]$_.n_away_venue -lt $summary.min_prior_venue_matches -or
    [double]$_.probability -lt 0 -or [double]$_.probability -gt 1 -or
    [double]$_.stress_probability -gt [double]$_.probability -or
    [int]$_.actual_under_7_5 -ne [int]([int]$_.total_1h -le 7)
})
if ($bad.Count) { throw 'Prediction boundary, threshold or stress check failed.' }
foreach ($day in ($top | Group-Object match_date)) {
    if ($day.Count -gt 3 -or @($day.Group.daily_rank | Sort-Object -Unique).Count -ne $day.Count) {
        throw 'Invalid daily top3.'
    }
    $expected = @($pred | Where-Object { $_.match_date -eq $day.Name } |
        Sort-Object @{Expression={[math]::Round([double]$_.probability,12)};Descending=$true},league,home_team,away_team |
        Select-Object -First 3)
    $actual = @($day.Group | Sort-Object { [int]$_.daily_rank })
    for ($i=0; $i -lt $actual.Count; $i++) {
        if ($actual[$i].league -ne $expected[$i].league -or
                $actual[$i].home_team -ne $expected[$i].home_team -or
                $actual[$i].away_team -ne $expected[$i].away_team) {
            throw "Daily ranking mismatch: $($day.Name), rank $($i+1); actual=$($actual[$i].league)/$($actual[$i].home_team)/$($actual[$i].probability); expected=$($expected[$i].league)/$($expected[$i].home_team)/$($expected[$i].probability)."
        }
    }
}
$rejected = @{}
Import-Csv -LiteralPath (Join-Path $folder 'corners_1h_rejected_rows.csv') | ForEach-Object { $rejected[$_.event_id]=$true }
$history = @(Import-Csv -LiteralPath (Join-Path $root 'simulations/Corners 1H/data/corners_1h_history.csv') |
    Where-Object { -not $rejected.ContainsKey($_.event_id) -and $_.match_date -lt $AsOf })
$indices = @(0,1,[int]($pred.Count/4),[int]($pred.Count/2),[int](3*$pred.Count/4),($pred.Count-1)) | Sort-Object -Unique
foreach ($index in $indices) {
    $r=$pred[$index]
    $prior=@($history | Where-Object {
        $_.league -eq $r.league -and $_.season -eq $r.season -and $_.match_date -lt $r.match_date
    })
    $h=@($prior | Where-Object { $_.home_team -eq $r.home_team -or $_.away_team -eq $r.home_team })
    $a=@($prior | Where-Object { $_.home_team -eq $r.away_team -or $_.away_team -eq $r.away_team })
    $hv=@($h | Where-Object { $_.home_team -eq $r.home_team })
    $av=@($a | Where-Object { $_.away_team -eq $r.away_team })
    if ($h.Count -ne [int]$r.n_home_season -or $a.Count -ne [int]$r.n_away_season -or
            $hv.Count -ne [int]$r.n_home_venue -or $av.Count -ne [int]$r.n_away_venue) {
        throw "Strictly-prior sample check failed for $($r.match_date) $($r.home_team)."
    }
}
[pscustomobject]@{status='PASS';input_hashes_verified=$manifest.Count;
    predictions_checked=$pred.Count;daily_rankings_checked=@($top | Group-Object match_date).Count;
    strictly_prior_sample_checks=$indices.Count} | ConvertTo-Json
