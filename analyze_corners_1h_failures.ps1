param([string]$AsOf='2026-11-01', [string]$CalibrationCutoff='2025-07-01')
$ErrorActionPreference='Stop'
$folder=Join-Path $PSScriptRoot "Analysis/Football70_Corners1H_Audit_$AsOf"
$culture=[cultureinfo]::InvariantCulture
function Num($Value) { return [double]::Parse([string]$Value,$culture) }
function Sigmoid([double]$Z) {
    if ($Z -ge 0) { return 1.0/(1.0+[math]::Exp(-$Z)) }
    $e=[math]::Exp($Z); return $e/(1.0+$e)
}
function Fit-Stats($Rows,[double]$A,[double]$B) {
    $loss=0.0; $ga=0.0; $gb=0.0; $haa=0.0; $hab=0.0; $hbb=0.0
    foreach ($r in $Rows) {
        $z=$A+$B*$r.x; $p=Sigmoid $z; $w=$p*(1-$p); $error=$p-$r.y
        $softplus=if ($z -gt 0) { $z+[math]::Log(1+[math]::Exp(-$z)) } else { [math]::Log(1+[math]::Exp($z)) }
        $loss+=$softplus-$r.y*$z; $ga+=$error; $gb+=$error*$r.x
        $haa+=$w; $hab+=$w*$r.x; $hbb+=$w*$r.x*$r.x
    }
    $ridge=1e-6
    [pscustomobject]@{loss=$loss+$ridge*($A*$A+($B-1)*($B-1))/2;
        ga=$ga+$ridge*$A;gb=$gb+$ridge*($B-1);haa=$haa+$ridge;hab=$hab;hbb=$hbb+$ridge}
}
function Metrics($Rows,[string]$Scope,[string]$Variant,[string]$ProbabilityField) {
    $n=@($Rows).Count
    if (-not $n) { return }
    $hits=0; $ps=0.0; $bs=0.0; $ll=0.0
    foreach ($r in $Rows) {
        $y=[int]$r.actual_under_7_5; $p=[double]$r.$ProbabilityField
        $q=[math]::Max(1e-12,[math]::Min(1-1e-12,$p))
        $hits+=$y; $ps+=$p; $bs+=($p-$y)*($p-$y)
        $ll-=$y*[math]::Log($q)+(1-$y)*[math]::Log(1-$q)
    }
    [pscustomobject]@{scope=$Scope;variant=$Variant;n=$n;hits=$hits;failure_rate=1-$hits/[double]$n;
        hit_rate=$hits/[double]$n;mean_probability=$ps/$n;brier=$bs/$n;log_loss=$ll/$n}
}
$all=@(Import-Csv -LiteralPath (Join-Path $folder 'corners_1h_brentford_walkforward_predictions.csv'))
$ranked=@(Import-Csv -LiteralPath (Join-Path $folder 'corners_1h_daily_top3_retrospective.csv'))
$top1=@($ranked | Where-Object { $_.daily_rank -eq '1' })
$failures=@($top1 | Where-Object { $_.actual_under_7_5 -eq '0' })
$failures | Export-Csv -LiteralPath (Join-Path $folder 'top1_failure_matches.csv') -NoTypeInformation -Encoding UTF8
$totals=@($failures | Group-Object total_1h | Sort-Object {[int]$_.Name} | ForEach-Object {
    [pscustomobject]@{total_1h=[int]$_.Name;failures=$_.Count;share_of_failures=$_.Count/[double]$failures.Count}
})
$totals | Export-Csv -LiteralPath (Join-Path $folder 'top1_failure_totals.csv') -NoTypeInformation -Encoding UTF8
$leagueRows=@($top1 | Group-Object league | Sort-Object Name | ForEach-Object {
    Metrics $_.Group $_.Name 'raw_poisson' 'probability'
})
$leagueRows | Export-Csv -LiteralPath (Join-Path $folder 'top1_failure_by_league.csv') -NoTypeInformation -Encoding UTF8
$sampleRows=@($top1 | Group-Object {
    $n=[math]::Min([int]$_.n_home_venue,[int]$_.n_away_venue)
    if ($n -lt 5) { '3-4' } elseif ($n -lt 10) { '5-9' } else { '10plus' }
} | ForEach-Object { Metrics $_.Group $_.Name 'raw_poisson' 'probability' })
$sampleRows | Export-Csv -LiteralPath (Join-Path $folder 'top1_failure_by_prior_venue_sample.csv') -NoTypeInformation -Encoding UTF8

# One declared chronological split, no cutoff/parameter search on the evaluation set.
$train=@($all | Where-Object { $_.match_date -lt $CalibrationCutoff } | ForEach-Object {
    $p=Num $_.probability; $p=[math]::Max(1e-12,[math]::Min(1-1e-12,$p))
    [pscustomobject]@{x=[math]::Log($p/(1-$p));y=[int]$_.actual_under_7_5;date=$_.match_date}
})
$validation=@($all | Where-Object { $_.match_date -ge $CalibrationCutoff -and $_.match_date -lt $AsOf })
if ($train.Count -lt 100 -or $validation.Count -lt 100) { throw 'Insufficient chronological split.' }
$a=0.0; $b=1.0; $converged=$false; $iterations=0
for ($iteration=0; $iteration -lt 50; $iteration++) {
    $iterations++
    $s=Fit-Stats $train $a $b
    $det=$s.haa*$s.hbb-$s.hab*$s.hab
    if ($det -le 1e-12) { throw 'Singular calibration fit.' }
    $da=($s.hbb*$s.ga-$s.hab*$s.gb)/$det
    $db=($s.haa*$s.gb-$s.hab*$s.ga)/$det
    if ([math]::Max([math]::Abs($da),[math]::Abs($db)) -lt 1e-9) { $converged=$true; break }
    $step=1.0; $accepted=$false
    for ($line=0; $line -lt 25; $line++) {
        $candidateA=$a-$step*$da; $candidateB=$b-$step*$db
        $candidate=Fit-Stats $train $candidateA $candidateB
        if ($candidate.loss -le $s.loss+1e-10) {
            $a=$candidateA; $b=$candidateB; $accepted=$true; break
        }
        $step/=2
    }
    if (-not $accepted) { throw 'Calibration line search failed.' }
}
if (-not $converged) { throw 'Calibration did not converge.' }
if ($b -le 0) { throw 'Non-monotonic calibration: do not reuse for unchanged ranking.' }
$calibrated=foreach ($r in $validation) {
    $p=Num $r.probability
    $p=[math]::Max(1e-12,[math]::Min(1-1e-12,$p))
    $r | Add-Member -NotePropertyName calibrated_probability -NotePropertyValue (Sigmoid ($a+$b*[math]::Log($p/(1-$p))))
    $r
}
$top1Keys=@{}
foreach ($r in $top1) { $top1Keys["$($r.match_date)|$($r.league)|$($r.home_team)|$($r.away_team)"]=$true }
$validationTop1=@($calibrated | Where-Object { $top1Keys.ContainsKey("$($_.match_date)|$($_.league)|$($_.home_team)|$($_.away_team)") })
$comparison=@(
    (Metrics $calibrated 'later_period_all_eligible' 'raw_poisson' 'probability'),
    (Metrics $calibrated 'later_period_all_eligible' 'platt_trained_on_earlier_period' 'calibrated_probability'),
    (Metrics $validationTop1 'later_period_same_top1_picks' 'raw_poisson' 'probability'),
    (Metrics $validationTop1 'later_period_same_top1_picks' 'platt_trained_on_earlier_period' 'calibrated_probability')
)
$comparison | Export-Csv -LiteralPath (Join-Path $folder 'calibration_chronological_comparison.csv') -NoTypeInformation -Encoding UTF8
$calibrated | Export-Csv -LiteralPath (Join-Path $folder 'calibration_later_period_predictions.csv') -NoTypeInformation -Encoding UTF8
$summary=[ordered]@{as_of=$AsOf;calibration_cutoff=$CalibrationCutoff;train_n=$train.Count;later_period_n=$validation.Count;
    fit_intercept=$a;fit_slope=$b;fit_converged=$converged;fit_iterations=$iterations;
    top1_n=$top1.Count;top1_failures=$failures.Count;top1_failure_rate=$failures.Count/[double]$top1.Count;
    failure_totals=$totals;comparison=$comparison;
    historical_outcomes_unchanged=$true;production_calibration_changed=$false;
    warning='Exploratory chronological correction after inspecting the overall historical audit. Not a prospective untouched final validation; no claim of hit-rate or ranking improvement.'}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $folder 'failure_analysis_summary.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 6
