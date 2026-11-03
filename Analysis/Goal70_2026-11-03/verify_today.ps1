param([string]$RecordedAtUtc)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$models=@(Import-Csv (Join-Path $root 'all_models.csv') -Encoding UTF8)
$h=@(Import-Csv (Join-Path $root 'Data/historical_validated.csv') -Encoding UTF8)
$f=@(Get-Content (Join-Path $root 'candidate_fixtures.json') -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
$ci=[Globalization.CultureInfo]::InvariantCulture
function Num($v){[double]::Parse([string]$v,$ci)}
function Check($ok,[string]$message){if(-not $ok){throw $message}}
function Minute([string]$v){Check ($v -match '^(\d+)') ('Bad minute '+$v);[int]$Matches[1]}
function Sample($x,[string]$side,[string]$venue='all'){
 $league=$x.($side+'_league');$id=$x.($side+'_history_id')
 @($h|Where-Object {$_.league -eq $league -and (($venue -eq 'all' -and ($_.home_id -eq $id -or $_.away_id -eq $id)) -or ($venue -eq 'home' -and $_.home_id -eq $id) -or ($venue -eq 'away' -and $_.away_id -eq $id))})
}
function Stats($rows,[string]$id){
 $gf=0;$ga=0
 foreach($r in $rows){if($r.home_id -eq $id){$gf += [int]$r.home70;$ga += [int]$r.away70}else{$gf += [int]$r.away70;$ga += [int]$r.home70}}
 [pscustomobject]@{n=@($rows).Count;gf=$gf;ga=$ga}
}
Check ($models.Count -eq 20) 'Expected 20 candidates'
Check (@($models|Group-Object competition,id|Where-Object Count -gt 1).Count -eq 0) 'Duplicate candidates'
Check (@($h|Group-Object league,id|Where-Object Count -gt 1).Count -eq 0) 'Duplicate history'
foreach($r in $h){
 Check ([DateTimeOffset]::Parse($r.date) -lt [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00')) ('Future history '+$r.id)
 $hg=@($r.home_goals -split ';'|Where-Object {$_});$ag=@($r.away_goals -split ';'|Where-Object {$_})
 Check ($hg.Count -eq [int]$r.home_score -and $ag.Count -eq [int]$r.away_score) ('FT mismatch '+$r.id)
 $a=@($hg|Where-Object {(Minute $_) -le 70}).Count;$b=@($ag|Where-Object {(Minute $_) -le 70}).Count
 Check ($a -eq [int]$r.home70 -and $b -eq [int]$r.away70) ('Goal70 mismatch '+$r.id)
 Check (($a+$b -gt 0) -eq ($r.success70 -eq 'True')) ('Label mismatch '+$r.id)
}
$tz=[TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
$asof=[DateTimeOffset]::Parse($RecordedAtUtc)
$rows=New-Object System.Collections.Generic.List[object]
foreach($x in $models){
 $a=Stats @(Sample $x 'home') $x.home_history_id;$b=Stats @(Sample $x 'away') $x.away_history_id
 $av=Stats @(Sample $x 'home' 'home') $x.home_history_id;$bv=Stats @(Sample $x 'away' 'away') $x.away_history_id
 foreach($pair in @(@('home',$a),@('away',$b),@('home_venue',$av),@('away_venue',$bv))){foreach($z in @(@('n','n'),@('gf70','gf'),@('ga70','ga'))){Check ([int]$x.($pair[0]+'_'+$z[0]) -eq [int]$pair[1].($z[1])) ('Input mismatch '+$x.id)}}
 $ls=($a.gf/$a.n+$a.ga/$a.n+$b.gf/$b.n+$b.ga/$b.n)/2
 $lv=($av.gf/$av.n+$av.ga/$av.n+$bv.gf/$bv.n+$bv.ga/$bv.n)/2
 $lc=($ls+$lv)/2;$p=1-[math]::Exp(-$lc);$stress=1-[math]::Exp(-.8*$lc)
 Check ([math]::Abs($p-(Num $x.p_raw)) -lt 1e-12 -and [math]::Abs($stress-(Num $x.p_stress)) -lt 1e-12) ('Probability mismatch '+$x.id)
 $fixture=@($f|Where-Object {$_.id -eq $x.id -and $_.competition -eq $x.competition})
 Check ($fixture.Count -eq 1 -and $fixture[0].state -eq 'pre' -and $fixture[0].time_valid) ('Not pre match '+$x.id)
 Check ([DateTimeOffset]::Parse($x.date) -gt $asof) ('Started '+$x.id)
 $u=@(@(Sample $x 'home')+@(Sample $x 'away')|Sort-Object league,id -Unique)
 Check ($u.Count -eq [int]$x.history_unique_n -and @($u|Where-Object success70 -eq 'True').Count -eq [int]$x.history_success) ('Duplicate adjusted frequency mismatch '+$x.id)
 $rows.Add([pscustomobject]@{rank=$rows.Count+1;competition=$x.competition;id=$x.id;home=$x.home;away=$x.away;kickoff_ro=[TimeZoneInfo]::ConvertTime([DateTimeOffset]::Parse($x.date),$tz).ToString('HH:mm');p_raw=$p;p_stress=$stress;fair_odds=1/$p;season_only=1-[math]::Exp(-$ls);venue_only=1-[math]::Exp(-$lv);first_goal_success=$x.history_success;first_goal_n=$x.history_unique_n;venue_success=$x.venue_success;venue_n=$x.venue_unique_n;history_basis=$x.history_basis;analyst_selected=($x.id -eq '401880179');user_selected=$false;first_goal_actual=$null;outcome=$null;recorded_at_ro=[TimeZoneInfo]::ConvertTime($asof,$tz).ToString('o');source=$x.source})
}
$rows|Export-Csv (Join-Path $root 'comparison_verified.csv') -NoTypeInformation -Encoding UTF8
# Independent provider reconciliation for both league pairs, including all season observations.
$cross=New-Object System.Collections.Generic.List[object]
foreach($pair in @(@('qpr','eng.2','334','10172'),@('burnley','eng.2','379','8191'),@('sirius','swe.1','13010','6694'),@('halmstad','swe.1','2978','8310'))){
 $fm=@(Get-Content (Join-Path $root ('Finalists/'+$pair[0]+'_league.json')) -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
 # IDs are reconciled from the candidate crosswalk, rather than fuzzy team names.
 $model=@($models|Where-Object {$_.home -match $(switch($pair[0]){'qpr'{'Queens Park'}'sirius'{'Sirius'}default{'__none__'}}) -or $_.away -match $(switch($pair[0]){'burnley'{'^Burnley$'}'halmstad'{'Halmstads'}default{'__none__'}})})[0]
 $side=if($pair[0] -in @('qpr','sirius')){'home'}else{'away'}
 $es=@(Sample $model $side)
 Check ($es.Count -eq $fm.Count) ('Season coverage difference '+$pair[0])
 foreach($r in $es){
  $date=([DateTimeOffset]::Parse($r.date)).UtcDateTime.ToString('yyyy-MM-dd')
  $match=@($fm|Where-Object {([DateTimeOffset]::Parse($_.date)).UtcDateTime.ToString('yyyy-MM-dd') -eq $date})
  Check ($match.Count -eq 1) ('Match date conflict '+$pair[0]+' '+$date)
  $m=$match[0];$score=([int]$r.home_score).ToString()+' - '+([int]$r.away_score).ToString()
  Check ($score -eq $m.score) ('Provider FT conflict '+$r.id)
  $fh=@($m.goals|Where-Object {$_.side -eq 'home' -and $_.minute -le 70}).Count;$fa=@($m.goals|Where-Object {$_.side -eq 'away' -and $_.minute -le 70}).Count
  Check ($fh -eq [int]$r.home70 -and $fa -eq [int]$r.away70) ('Provider Goal70 conflict '+$r.id)
  $red=@($m.cards|Where-Object {$_.card -match 'Red'})
  $cross.Add([pscustomobject]@{team=$pair[0];date=$date;espn_id=$r.id;fotmob_id=$m.id;home=$r.home;away=$r.away;home70=$fh;away70=$fa;first_goal=$m.first_goal;red_cards=$red.Count;red_minutes=($red|ForEach-Object {$_.minute}) -join ';';source=$m.source})
 }
}
$cross|Export-Csv (Join-Path $root 'independent_provider_checks.csv') -NoTypeInformation -Encoding UTF8
# Diagnostics only; the original prediction is never replaced by these exclusions.
$sens=New-Object System.Collections.Generic.List[object]
foreach($spec in @(@('QPR-Burnley','401880179','qpr','burnley'),@('Sirius-Halmstad','401069555','sirius','halmstad'))){
 $x=@($models|Where-Object {($_.id -eq $spec[1]) -or ($spec[0] -eq 'Sirius-Halmstad' -and $_.home -match 'Sirius')})[0]
 foreach($variant in @('original','exclude_red_card_matches','exclude_qpr_5_0_swansea')){
  if($variant -eq 'exclude_qpr_5_0_swansea' -and $spec[0] -ne 'QPR-Burnley'){continue}
  $aRows=@(Sample $x 'home');$bRows=@(Sample $x 'away')
  $badIds=if($variant -eq 'exclude_red_card_matches'){@($cross|Where-Object {$_.team -in @($spec[2],$spec[3]) -and [int]$_.red_cards -gt 0}|Select-Object -ExpandProperty espn_id)}elseif($variant -eq 'exclude_qpr_5_0_swansea'){@($cross|Where-Object {$_.team -eq 'qpr' -and $_.date -eq '2026-10-24'}|Select-Object -ExpandProperty espn_id)}else{@()}
  $aRows=@($aRows|Where-Object {$_.id -notin $badIds});$bRows=@($bRows|Where-Object {$_.id -notin $badIds})
  $a=Stats $aRows $x.home_history_id;$b=Stats $bRows $x.away_history_id
  $av=Stats @($aRows|Where-Object home_id -eq $x.home_history_id) $x.home_history_id;$bv=Stats @($bRows|Where-Object away_id -eq $x.away_history_id) $x.away_history_id
  $ls=($a.gf/$a.n+$a.ga/$a.n+$b.gf/$b.n+$b.ga/$b.n)/2;$lv=($av.gf/$av.n+$av.ga/$av.n+$bv.gf/$bv.n+$bv.ga/$bv.n)/2
  $sens.Add([pscustomobject]@{match=$spec[0];variant=$variant;home_n=$a.n;away_n=$b.n;home_venue_n=$av.n;away_venue_n=$bv.n;excluded_ids=$badIds -join ';';lambda_season=$ls;lambda_venue=$lv;p=1-[math]::Exp(-($ls+$lv)/2)})
 }
}
$sens|Export-Csv (Join-Path $root 'sensitivity.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{recorded_at_utc=$RecordedAtUtc;models=$models.Count;historical_validated=$h.Count;historical_rejected=@(Import-Csv (Join-Path $root 'Data/historical_rejected.csv')).Count;independent_team_observations=$cross.Count;all_checks_passed=$true;future_results_present=$false}|ConvertTo-Json|Set-Content (Join-Path $root 'verification.json') -Encoding UTF8
$sens|ConvertTo-Json
