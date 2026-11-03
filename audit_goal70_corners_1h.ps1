param(
    [string]$AsOf = '2026-11-01',
    [int]$MinSeasonMatches = 5,
    [int]$MinVenueMatches = 3
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$cutoff = [datetime]::ParseExact($AsOf, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture)
if ($MinSeasonMatches -lt 1 -or $MinVenueMatches -lt 1) { throw 'Sample thresholds must be positive.' }
$outDir = Join-Path $root "Analysis/Football70_Corners1H_Audit_$AsOf"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

function Read-Number($Value) {
    $number = 0.0
    if ([string]::IsNullOrWhiteSpace([string]$Value)) { return $null }
    if ([double]::TryParse([string]$Value, [globalization.numberstyles]::Float,
            [cultureinfo]::InvariantCulture, [ref]$number) -and
            -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)) { return $number }
    return $null
}

function Under-Probability([double]$Mu) {
    if ($Mu -lt 0 -or [double]::IsNaN($Mu)) { throw 'Invalid intensity.' }
    $term = [math]::Exp(-$Mu)
    $sum = $term
    for ($k = 1; $k -le 7; $k++) { $term *= $Mu / $k; $sum += $term }
    return $sum
}

function New-TeamState {
    return @{ n=0; gf=0.0; ga=0.0; nh=0; gfh=0.0; gah=0.0; na=0; gfa=0.0; gaa=0.0 }
}

function Write-Csv($Rows, [string]$Name) {
    $Rows | Export-Csv -LiteralPath (Join-Path $outDir $Name) -NoTypeInformation -Encoding UTF8
}

function Get-Metrics($Rows, [string]$Scope, [string]$League) {
    $n = @($Rows).Count
    if ($n -eq 0) {
        return [pscustomobject]@{scope=$Scope; league=$League; n=0; hits=0;
            hit_rate=$null; mean_probability=$null; brier=$null; log_loss=$null}
    }
    $hits=0; $pSum=0.0; $brier=0.0; $logLoss=0.0
    foreach ($r in $Rows) {
        $y=[int]$r.actual_under_7_5; $p=[double]$r.probability
        $hits += $y; $pSum += $p; $brier += [math]::Pow($p-$y, 2)
        $safe = [math]::Max(1e-12, [math]::Min(1-1e-12, $p))
        $logLoss -= $y*[math]::Log($safe) + (1-$y)*[math]::Log(1-$safe)
    }
    return [pscustomobject]@{scope=$Scope; league=$League; n=$n; hits=$hits;
        hit_rate=$hits/[double]$n; mean_probability=$pSum/$n; brier=$brier/$n; log_loss=$logLoss/$n}
}

# Deterministic boundary checks; these are mathematical checks, not fitted tests.
if ([math]::Abs((Under-Probability 0)-1) -gt 1e-12) { throw 'Poisson zero-intensity check failed.' }
if ((Under-Probability 5) -ge (Under-Probability 4)) { throw 'Under probability must fall as intensity rises.' }
if ([math]::Abs((Under-Probability 3.5208333333333335)-0.9724495796110346) -gt 1e-10) {
    throw 'Reference Poisson calculation failed.'
}

$sourcePaths = @(
    'simulations/Corners 1H/data/corners_1h_history.csv',
    'simulations/Corners 1H/backtests/corners_1h_predictions.csv',
    'simulations/FHG/data/fhg_history.csv',
    'data/flashscore/match_stats.csv',
    'football-goal70-2026-10-14.csv'
)
$manifest = foreach ($relative in $sourcePaths) {
    $path = Join-Path $root $relative
    $file = Get-Item -LiteralPath $path
    [pscustomobject]@{path=$relative; bytes=$file.Length;
        sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash;
        header=(Get-Content -LiteralPath $path -TotalCount 1 -Encoding UTF8)}
}
Write-Csv $manifest 'source_manifest.csv'

$raw = @(Import-Csv -LiteralPath (Join-Path $root $sourcePaths[0]))
$issues = [collections.generic.list[object]]::new()
$byFixture = @{}
$idToFixture = @{}
$conflicts = [collections.generic.hashset[string]]::new()
$duplicateRows = 0
foreach ($r in $raw) {
    $date = [datetime]::MinValue
    $h = Read-Number $r.home_corners_1h; $a = Read-Number $r.away_corners_1h
    $reason = $null
    if (-not [datetime]::TryParseExact($r.match_date, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture,
            [globalization.datetimestyles]::None, [ref]$date)) { $reason = 'invalid_date' }
    elseif ($date -ge $cutoff) { $reason = 'on_or_after_cutoff' }
    elseif ($null -eq $h -or $null -eq $a) { $reason = 'missing_or_non_numeric_1h_corners' }
    elseif ($h -lt 0 -or $a -lt 0 -or $h -ne [math]::Floor($h) -or $a -ne [math]::Floor($a)) {
        $reason = 'negative_or_fractional_1h_corners'
    }
    elseif ([string]::IsNullOrWhiteSpace($r.league) -or [string]::IsNullOrWhiteSpace($r.season) -or
            [string]::IsNullOrWhiteSpace($r.home_team) -or [string]::IsNullOrWhiteSpace($r.away_team)) {
        $reason = 'missing_fixture_identity'
    }
    if (-not $reason) {
        foreach ($side in @('home','away')) {
            $first = Read-Number $r."${side}_corners_1h"
            $second = Read-Number $r."${side}_corners_2h"
            $total = Read-Number $r."${side}_corners_total"
            if ($null -ne $total -and ($total -lt $first -or
                    ($null -ne $second -and $first+$second -ne $total))) {
                $reason = 'half_full_time_corner_mismatch'
            }
        }
    }
    if ($reason) {
        $issues.Add([pscustomobject]@{league=$r.league;match_date=$r.match_date;event_id=$r.event_id;reason=$reason})
        continue
    }
    $league=$r.league.Trim().ToUpperInvariant()
    $homeName=($r.home_team.Trim().ToLowerInvariant() -replace '\s+',' ')
    $away=($r.away_team.Trim().ToLowerInvariant() -replace '\s+',' ')
    $key="$league|$($r.match_date)|$homeName|$away"
    if ($r.event_id -and $idToFixture.ContainsKey($r.event_id) -and $idToFixture[$r.event_id] -ne $key) {
        [void]$conflicts.Add($key); [void]$conflicts.Add($idToFixture[$r.event_id])
    }
    if ($r.event_id) { $idToFixture[$r.event_id]=$key }
    if ($byFixture.ContainsKey($key)) {
        $duplicateRows++
        if ($byFixture[$key].h -ne $h -or $byFixture[$key].a -ne $a -or $byFixture[$key].season -ne $r.season) {
            [void]$conflicts.Add($key)
        }
        continue
    }
    $byFixture[$key]=[pscustomobject]@{league=$league;season=$r.season;match_date=$r.match_date;
        home_team=$homeName;away_team=$away;h=$h;a=$a;total=$h+$a;event_id=$r.event_id;source=$r.source}
}
$valid = @($byFixture.GetEnumerator() | Where-Object { -not $conflicts.Contains($_.Key) } |
    ForEach-Object { $_.Value } | Sort-Object match_date,league,home_team,away_team)
$coverage = foreach ($g in ($valid | Group-Object league | Sort-Object Name)) {
    $dates=@($g.Group.match_date | Sort-Object)
    $success=@($g.Group | Where-Object { $_.total -le 7 }).Count
    [pscustomobject]@{league=$g.Name;valid_unique_matches=$g.Count;
        seasons=@($g.Group.season | Sort-Object -Unique).Count;first_date=$dates[0];last_date=$dates[-1];
        days_stale=($cutoff-[datetime]$dates[-1]).Days;under_7_5=$success;
        eight_or_more=$g.Count-$success;historical_under_rate=$success/[double]$g.Count;
        mean_corners_1h=($g.Group.total | Measure-Object -Average).Average}
}
Write-Csv $coverage 'corners_1h_coverage_by_league.csv'
if ($issues.Count) { Write-Csv $issues 'corners_1h_rejected_rows.csv' }

# Each date is predicted before any result from that date updates any team state.
# State is isolated by league and season. No odds or calibrated production artifacts are used.
$states=@{}; $predictions=[collections.generic.list[object]]::new()
$skipped=[collections.generic.list[object]]::new()
foreach ($day in ($valid | Group-Object match_date | Sort-Object Name)) {
    foreach ($m in $day.Group) {
        $hk="$($m.league)|$($m.season)|$($m.home_team)"
        $ak="$($m.league)|$($m.season)|$($m.away_team)"
        if (-not $states.ContainsKey($hk)) { $states[$hk]=New-TeamState }
        if (-not $states.ContainsKey($ak)) { $states[$ak]=New-TeamState }
        $hs=$states[$hk]; $as=$states[$ak]
        if ($hs.n -lt $MinSeasonMatches -or $as.n -lt $MinSeasonMatches -or
                $hs.nh -lt $MinVenueMatches -or $as.na -lt $MinVenueMatches) {
            $skipped.Add([pscustomobject]@{match_date=$m.match_date;league=$m.league;
                home_team=$m.home_team;away_team=$m.away_team;reason='insufficient_prior_season_or_venue_sample'})
            continue
        }
        $ls=($hs.gf/$hs.n + $as.ga/$as.n + $as.gf/$as.n + $hs.ga/$hs.n)/2
        $lv=($hs.gfh/$hs.nh + $as.gaa/$as.na + $as.gfa/$as.na + $hs.gah/$hs.nh)/2
        $lambda=($ls+$lv)/2
        $p=Under-Probability $lambda
        $predictions.Add([pscustomobject]@{match_date=$m.match_date;league=$m.league;season=$m.season;
            home_team=$m.home_team;away_team=$m.away_team;n_home_season=$hs.n;n_away_season=$as.n;
            n_home_venue=$hs.nh;n_away_venue=$as.na;lambda_season=$ls;lambda_venue=$lv;lambda_combined=$lambda;
            probability=$p;stress_probability=(Under-Probability ($lambda*1.2));fair_odds=1/$p;
            total_1h=$m.total;actual_under_7_5=[int]($m.total -le 7)})
    }
    foreach ($m in $day.Group) {
        $hs=$states["$($m.league)|$($m.season)|$($m.home_team)"]
        $as=$states["$($m.league)|$($m.season)|$($m.away_team)"]
        $hs.n++; $hs.gf+=$m.h; $hs.ga+=$m.a; $hs.nh++; $hs.gfh+=$m.h; $hs.gah+=$m.a
        $as.n++; $as.gf+=$m.a; $as.ga+=$m.h; $as.na++; $as.gfa+=$m.a; $as.gaa+=$m.h
    }
}
if ($predictions.Count) { Write-Csv $predictions 'corners_1h_brentford_walkforward_predictions.csv' }
if ($skipped.Count) { Write-Csv $skipped 'corners_1h_insufficient_history.csv' }
$metrics=[collections.generic.list[object]]::new()
$metrics.Add((Get-Metrics $predictions 'all_eligible' 'ALL'))
foreach ($g in ($predictions | Group-Object league | Sort-Object Name)) {
    $metrics.Add((Get-Metrics $g.Group 'all_eligible' $g.Name))
}
$topRows=[collections.generic.list[object]]::new()
foreach ($g in ($predictions | Group-Object match_date)) {
    $ordered=@($g.Group | Sort-Object @{Expression={[math]::Round([double]$_.probability,12)};Descending=$true},league,home_team,away_team)
    for ($i=0; $i -lt [math]::Min(3,$ordered.Count); $i++) {
        $r=$ordered[$i] | Select-Object *
        $r | Add-Member -NotePropertyName daily_rank -NotePropertyValue ($i+1)
        $topRows.Add($r)
    }
}
if ($topRows.Count) { Write-Csv $topRows 'corners_1h_daily_top3_retrospective.csv' }
$metrics.Add((Get-Metrics @($topRows | Where-Object { $_.daily_rank -eq 1 }) 'daily_top1' 'ALL'))
$metrics.Add((Get-Metrics $topRows 'daily_top3' 'ALL'))
Write-Csv $metrics 'corners_1h_brentford_walkforward_metrics.csv'
$buckets=foreach ($g in ($predictions | Group-Object { [math]::Min(19,[int][math]::Floor($_.probability*20)) } | Sort-Object { [int]$_.Name })) {
    $r=Get-Metrics $g.Group 'probability_bucket' 'ALL'
    $r | Add-Member -NotePropertyName bucket_lower -NotePropertyValue ([int]$g.Name/20)
    $r | Add-Member -NotePropertyName bucket_upper -NotePropertyValue (([int]$g.Name+1)/20)
    $r
}
if (@($buckets).Count) { Write-Csv $buckets 'corners_1h_brentford_calibration_buckets.csv' }

$manual=@(Import-Csv -LiteralPath (Join-Path $root $sourcePaths[4]))
$manualIssues=[collections.generic.list[object]]::new()
foreach ($r in $manual) {
    $scored=@($r.scored_minutes -split ';' | Where-Object { $_.Trim() })
    $conceded=@($r.conceded_minutes -split ';' | Where-Object { $_.Trim() })
    $tokens=@($scored)+@($conceded)
    $reason=$null
    if ($scored.Count -ne (Read-Number $r.goals_for) -or $conceded.Count -ne (Read-Number $r.goals_against)) {
        $reason='goal_count_minute_list_mismatch'
    }
    elseif (@($tokens | Where-Object { $_ -notmatch '^\d{1,3}(\+\d+)?$' }).Count) { $reason='invalid_minute_token' }
    else {
        $first=if ($tokens.Count) { ($tokens | ForEach-Object { [int](($_ -split '\+')[0]) } | Measure-Object -Minimum).Minimum } else { $null }
        $hit=$null -ne $first -and $first -le 70
        if (($r.goal_by_70_displayed_minute -eq 'yes') -ne $hit) { $reason='goal70_label_minute_mismatch' }
    }
    if ($reason) { $manualIssues.Add([pscustomobject]@{team=$r.team;date=$r.date;match=$r.match;reason=$reason}) }
}
if ($manualIssues.Count) { Write-Csv $manualIssues 'goal70_manual_issues.csv' }
$manualCoverage=foreach ($g in ($manual | Group-Object team | Sort-Object Name)) {
    $dates=@($g.Group.date | Sort-Object)
    [pscustomobject]@{team=$g.Name;observations=$g.Count;first_date=$dates[0];last_date=$dates[-1]}
}
Write-Csv $manualCoverage 'goal70_manual_coverage.csv'
$fhg=@(Import-Csv -LiteralPath (Join-Path $root $sourcePaths[2]))
$fhgCoverage=foreach ($g in ($fhg | Group-Object league | Sort-Object Name)) {
    $dates=@($g.Group.match_date | Sort-Object)
    [pscustomobject]@{league=$g.Name;rows=$g.Count;first_date=$dates[0];last_date=$dates[-1];
        has_goal_minutes=$false;usable_for_goal70_exact_counts=$false}
}
Write-Csv $fhgCoverage 'fhg_coverage_not_goal70.csv'

$summary=[ordered]@{
    as_of=$AsOf;generated_at_utc=[datetime]::UtcNow.ToString('o');
    method='50pct_season_50pct_venue_poisson_baseline';ranking_probability_decimal_places=12;
    min_prior_season_matches=$MinSeasonMatches;min_prior_venue_matches=$MinVenueMatches;
    corners_raw_rows=$raw.Count;corners_valid_unique_rows=$valid.Count;
    corners_rejected_rows=$issues.Count;corners_duplicate_rows=$duplicateRows;corners_conflicting_fixture_keys=$conflicts.Count;
    corners_leagues=@($coverage).Count;corners_first_date=$valid[0].match_date;corners_last_date=$valid[-1].match_date;
    corners_backtest_predictions=$predictions.Count;corners_skipped_insufficient_history=$skipped.Count;
    corners_current_day_predictions=0;current_day_status='NOT_RUN_STALE_HISTORY_AND_NO_LIVE_FIXTURE_SCREEN';
    goal70_manual_observations=$manual.Count;goal70_manual_unique_matches=@($manual | Group-Object date,match).Count;
    goal70_manual_teams=@($manualCoverage).Count;goal70_manual_structural_issues=$manualIssues.Count;
    goal70_full_league_backtest_available=$false;goal70_missing_fields=@('league','season','verified_event_identity','complete_season_goal_events','period');
    fhg_rows=$fhg.Count;fhg_leagues=@($fhgCoverage).Count;
    existing_corners_calibration_warning='train_corners_1h.py fits Platt calibration on the same walk-forward predictions used to report calibrated metrics; those calibrated metrics are not independent out-of-sample validation.';
    limitations=@('No live data refreshed.','History coverage is partial and stale.','No independent provider audit performed.',
        'Exact team-name normalization only; aliases are not guessed.','No profit or ROI measured; no odds used.',
        'Daily historical top3 uses only matches with eligible observed history, not a complete reconstructed fixture list.',
        'Thresholds were fixed before evaluating this run; no parameter search or calibration was performed.')
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir 'summary.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 6
$metrics | Where-Object { $_.league -eq 'ALL' } | ConvertTo-Json -Depth 3
