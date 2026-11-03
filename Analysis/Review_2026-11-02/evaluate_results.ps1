$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$analysis=Split-Path $root -Parent
$predictions=@(Import-Csv (Join-Path $analysis 'Selection_History/2026-11-02_goal70_candidates.csv') -Encoding UTF8)
$cornerPred=@(Import-Csv (Join-Path $analysis 'Corners1H_2026-11-02/all_models.csv') -Encoding UTF8)
$culture=[Globalization.CultureInfo]::InvariantCulture
function Num($x){[double]::Parse([string]$x,$culture)}
function Check($condition,[string]$s){if(-not $condition){throw $s}}
$map=@{
 '401874963'='5749738';'401874494'='5739537';'401885403'='5887671';'401882824'='5868113';'401874971'='5749731';
 '401876419'='5802975';'401876702'='5803306';'401883123'='5868524';'401896813'='5943282';'401878762'='5795503';
 '401883144'='5868523';'401896739'='5943281';'401842867'='5107648';'401883146'='5868521';'401883147'='5868520';
 '401876703'='5803305';'401842870'='5107646';'401888260'='5904789';'401885407'='5887667'
}
$all=New-Object System.Collections.Generic.List[object]
foreach($p in $predictions){
 $fmId=if($map.ContainsKey($p.id)){$map[$p.id]}else{$p.id}
 $fmPred=@($cornerPred|Where-Object id -eq $fmId)
 Check ($fmPred.Count -eq 1 -and $fmPred[0].league -eq $p.league) ('Crosswalk mismatch '+$p.id)
 $r=Get-Content (Join-Path $root ('FotMob/'+$fmId+'.json')) -Raw -Encoding UTF8|ConvertFrom-Json
 Check ($r.ok -and [string]$r.general.homeTeam.id -eq $fmPred[0].home_id -and [string]$r.general.awayTeam.id -eq $fmPred[0].away_id) ('Result identity mismatch '+$p.id)
 $status='unresolved';$first='';$firstMinute=$null;$label=$null;$score='';$reason='';$cross='FotMob_only';$espnSource='';$espnFirst=''
 if($r.general.finished -ne $true -and [DateTimeOffset]::Parse($r.status.utcTime).ToString('yyyy-MM-dd') -ne '2026-11-02'){
  $status='postponed_not_scored';$reason='Rescheduled beyond original prediction date; no completed match';
 }elseif($r.general.finished -eq $true){
  $h=@($r.teams|Where-Object {[string]$_.id -eq $fmPred[0].home_id})[0]
  $a=@($r.teams|Where-Object {[string]$_.id -eq $fmPred[0].away_id})[0]
  Check ([string]$h.score -match '^\d+$' -and [string]$a.score -match '^\d+$') ('Missing score '+$fmId)
  Check (@($r.goals|Where-Object side -eq 'home').Count -eq [int]$h.score -and @($r.goals|Where-Object side -eq 'away').Count -eq [int]$a.score) ('Goal-score mismatch '+$fmId)
  Check (@($r.goals|Where-Object {[int]$_.minute -gt 90 -or [int]$_.minute -lt 1}).Count -eq 0) ('Invalid goal period '+$fmId)
  $score=([string]$h.score+'-'+[string]$a.score)
  $g=@($r.goals|Sort-Object {[int]$_.minute},{[int]$_.added_time}|Select-Object -First 1)
  if($g.Count){$firstMinute=[int]$g[0].minute;$first=[string]$firstMinute;if($g[0].added_time -gt 0){$first+='+'+[string]$g[0].added_time};$label=[int]($firstMinute -le 70);$reason=if($label -eq 1){'first_goal_by_70'}else{'first_goal_after_70'}}
  else{Check ($score -eq '0-0') ('No goals but nonzero score '+$fmId);$label=0;$reason='no_goal_in_completed_match'}
  $status=if($label -eq 1){'success'}else{'fail'}
  $espnFile=Join-Path $root ('ESPN/'+$p.league+'.json')
  if(Test-Path -LiteralPath $espnFile){
   $ed=Get-Content $espnFile -Raw -Encoding UTF8|ConvertFrom-Json
   $em=@($ed.events|Where-Object id -eq $p.id)
   if($em.Count -eq 1){
    Check ($em[0].completed -eq $true) ('ESPN match not final '+$p.id)
    $eh=@($em[0].teams|Where-Object homeAway -eq 'home')[0];$ea=@($em[0].teams|Where-Object homeAway -eq 'away')[0]
    Check (([string]$eh.score+'-'+[string]$ea.score) -eq $score) ('Provider score conflict '+$p.id)
    $eg=@($em[0].goals|Sort-Object {[int]([regex]::Match($_.minute,'^\d+').Value)}|Select-Object -First 1)
    if($eg.Count){$espnFirst=$eg[0].minute;$en=[int]([regex]::Match($espnFirst,'^\d+').Value);Check ([int]($en -le 70) -eq $label) ('Threshold conflict '+$p.id);$cross=if($en -eq $firstMinute){'ESPN_FotMob_first_minute_and_score_agree'}else{'ESPN_FotMob_same_outcome_minute_difference'}}else{Check ($score -eq '0-0') ('ESPN missing goals '+$p.id);$cross='ESPN_FotMob_0_0_agree'}
    $espnSource=$em[0].source
   }
  }
 }
 $prob=Num $p.p_raw
 $all.Add([pscustomobject]@{rank=[int]$p.rank;league=$p.league;home=$p.home;away=$p.away;prediction_id=$p.id;fotmob_id=$fmId;
  model_snapshot_at_ro=$p.model_snapshot_at_ro;p_raw=$prob;p_stress=Num $p.p_stress;season_only=Num $p.season_only;venue_only=Num $p.venue_only;
  historical_frequency=Num $p.first_goal_frequency;historical_n=[int]$p.first_goal_unique_n;venue_historical_success=[int]$p.venue_unique_success;venue_historical_n=[int]$p.venue_unique_n;
  user_selected=$p.user_selected;analyst_finalist=$p.analyst_finalist;score_ft=$score;first_goal_minute=$first;first_goal_base=$firstMinute;label=$label;result=$status;reason=$reason;
  brier=if($null -ne $label){[math]::Pow($prob-$label,2)}else{$null};log_loss=if($null -ne $label){-($label*[math]::Log($prob)+(1-$label)*[math]::Log(1-$prob))}else{$null};
  crosscheck=$cross;espn_first_goal=$espnFirst;result_source=$r.match_source;espn_source=$espnSource;verified_at_utc=$r.fetched_at_utc})
}
Check ($all.Count -eq 27) 'Expected 27 candidate outcomes'
$all|Sort-Object rank|Export-Csv (Join-Path $root 'goal70_outcomes.csv') -NoTypeInformation -Encoding UTF8
$played=@($all|Where-Object {$null -ne $_.label});$wins=@($played|Where-Object label -eq 1).Count
$expected=($played|Measure-Object p_raw -Sum).Sum
$baseline=$wins/$played.Count
$constantBrier=($played|ForEach-Object {[math]::Pow($baseline-$_.label,2)}|Measure-Object -Average).Average
$bands=foreach($n in @(1,3,5,10,27)){
 $sample=@($all|Sort-Object rank|Select-Object -First $n|Where-Object {$null -ne $_.label})
 [pscustomobject]@{top_n=$n;played=$sample.Count;wins=@($sample|Where-Object label -eq 1).Count;hit_rate=@($sample|Where-Object label -eq 1).Count/$sample.Count;mean_p=($sample|Measure-Object p_raw -Average).Average}
}
$bands|Export-Csv (Join-Path $root 'top_n_results.csv') -NoTypeInformation -Encoding UTF8
$metrics=[pscustomobject]@{predictions=27;played=$played.Count;success=$wins;fail=$played.Count-$wins;postponed=@($all|Where-Object result -eq 'postponed_not_scored').Count;
 hit_rate=$baseline;mean_predicted_probability=$expected/$played.Count;expected_successes=$expected;expected_failures=$played.Count-$expected;
 brier=($played|Measure-Object brier -Average).Average;log_loss=($played|Measure-Object log_loss -Average).Average;
 constant_same_day_fitted_brier_not_out_of_sample=$constantBrier;independent_calibration=$false;
 crosschecked_with_espn=@($played|Where-Object {$_.crosscheck -like 'ESPN*'}).Count;reviewed_at_utc=[DateTime]::UtcNow.ToString('o')}
$metrics|ConvertTo-Json|Set-Content (Join-Path $root 'metrics.json') -Encoding UTF8
$metrics|ConvertTo-Json
$all|Where-Object {$_.result -ne 'success'}|Select-Object rank,home,away,p_raw,score_ft,first_goal_minute,result,crosscheck|ConvertTo-Json
$bands|ConvertTo-Json
