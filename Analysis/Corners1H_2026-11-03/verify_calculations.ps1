param([string]$RecordedAtUtc)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$ci=[Globalization.CultureInfo]::InvariantCulture
function Num($v){[double]::Parse([string]$v,$ci)}
function Check($ok,[string]$message){if(-not $ok){throw $message}}
function Cdf([double]$lambda){$factorial=1;$sum=0;for($k=0;$k -le 7;$k++){if($k){$factorial*=$k};$sum += [math]::Pow($lambda,$k)/$factorial};[math]::Exp(-$lambda)*$sum}
$h=@(Import-Csv (Join-Path $root 'historical_corners_1h.csv') -Encoding UTF8)
$models=@(Import-Csv (Join-Path $root 'all_models.csv') -Encoding UTF8)
$coverage=@(Import-Csv (Join-Path $root 'coverage.csv') -Encoding UTF8)
$status=@(Get-Content (Join-Path $root 'current_status.json') -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
Check ($h.Count -eq 406 -and $models.Count -eq 20 -and $coverage.Count -eq 15 -and $status.Count -eq 20) 'Coverage totals changed'
Check (@($h|Group-Object league,id|Where-Object Count -gt 1).Count -eq 0) 'Duplicate history'
Check (@($models|Where-Object eligible -eq 'True').Count -eq 19) 'Eligible total changed'
Check (@($coverage|Where-Object {[int]$_.not_fetched -ne 0}).Count -eq 0) 'Incomplete collection'
foreach($r in $h){
 Check ($r.period -eq 'FirstHalf' -and [DateTimeOffset]::Parse($r.date) -lt [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00')) ('Invalid period or time '+$r.id)
 $total=[int]$r.home_corners_1h+[int]$r.away_corners_1h
 Check ($total -eq [int]$r.total_1h -and ($r.under_7_5 -eq 'True') -eq ($total -le 7) -and ($r.eight_plus -eq 'True') -eq ($total -ge 8) -and ($r.exactly_7 -eq 'True') -eq ($total -eq 7)) ('Label mismatch '+$r.id)
 foreach($side in @('home','away')){Check (([int]$r.($side+'_corners_1h')+[int]$r.($side+'_corners_2h')) -eq [int]$r.($side+'_corners_ft')) ('Period sum mismatch '+$r.id)}
}
function Sample($x,[string]$side,[string]$venue='all'){
 $slug=$x.($side+'_league');$id=$x.($side+'_id')
 @($h|Where-Object {$_.league -eq $slug -and (($venue -eq 'all' -and ($_.home_id -eq $id -or $_.away_id -eq $id)) -or ($venue -eq 'home' -and $_.home_id -eq $id) -or ($venue -eq 'away' -and $_.away_id -eq $id))})
}
function Stats($rows,[string]$id){
 $cf=0;$ca=0;foreach($r in $rows){if($r.home_id -eq $id){$cf += [int]$r.home_corners_1h;$ca += [int]$r.away_corners_1h}else{$cf += [int]$r.away_corners_1h;$ca += [int]$r.home_corners_1h}}
 [pscustomobject]@{n=@($rows).Count;cf=$cf;ca=$ca}
}
foreach($x in $models){
 $s=@($status|Where-Object id -eq $x.id)
 Check ($s.Count -eq 1 -and $s[0].started -ne $true -and $s[0].finished -ne $true -and $s[0].cancelled -ne $true -and [DateTimeOffset]::Parse($x.date) -eq [DateTimeOffset]::Parse($s[0].date) -and [DateTimeOffset]::Parse($x.date) -gt [DateTimeOffset]::Parse($RecordedAtUtc)) ('Status or kickoff mismatch '+$x.id)
 if($x.eligible -ne 'True'){Check ($x.id -eq '6106335' -and $x.p_raw -eq '' -and [int]$x.home_n -eq 0 -and [int]$x.home_expected -eq 11) 'Unexpected missing-data exclusion';continue}
 $ar=@(Sample $x 'home');$br=@(Sample $x 'away');$avr=@(Sample $x 'home' 'home');$bvr=@(Sample $x 'away' 'away')
 foreach($pair in @(@('home',$ar,$x.home_id),@('away',$br,$x.away_id),@('home_venue',$avr,$x.home_id),@('away_venue',$bvr,$x.away_id))){
  $v=Stats $pair[1] $pair[2]
  foreach($field in @('n','cf','ca')){Check ([int]$x.($pair[0]+'_'+$field) -eq $v.$field) ('Input mismatch '+$x.id+' '+$field)}
 }
 $a=Stats $ar $x.home_id;$b=Stats $br $x.away_id;$av=Stats $avr $x.home_id;$bv=Stats $bvr $x.away_id
 $ls=(($a.cf+$a.ca)/$a.n+($b.cf+$b.ca)/$b.n)/2;$lv=(($av.cf+$av.ca)/$av.n+($bv.cf+$bv.ca)/$bv.n)/2;$lc=($ls+$lv)/2
 Check ([math]::Abs((Cdf $lc)-(Num $x.p_raw)) -lt 1e-12 -and [math]::Abs((Cdf (1.2*$lc))-(Num $x.p_stress)) -lt 1e-12) ('CDF mismatch '+$x.id)
 Check ([math]::Abs(1/(Cdf $lc)-(Num $x.fair_odds)) -lt 1e-12) ('Fair odds mismatch '+$x.id)
 $u=@(@($ar)+@($br)|Sort-Object league,id -Unique);$uv=@(@($avr)+@($bvr)|Sort-Object league,id -Unique)
 Check ($u.Count -eq [int]$x.unique_n -and @($u|Where-Object eight_plus -eq 'True').Count -eq [int]$x.unique_8plus -and @($u|Where-Object exactly_7 -eq 'True').Count -eq [int]$x.unique_7) ('Historical frequency mismatch '+$x.id)
 Check ($uv.Count -eq [int]$x.venue_unique_n -and @($uv|Where-Object eight_plus -eq 'True').Count -eq [int]$x.venue_8plus) ('Venue frequency mismatch '+$x.id)
}
$checks=@(Get-ChildItem (Join-Path $root 'ESPN') -Filter 'match_*.json'|ForEach-Object {Get-Content $_.FullName -Raw -Encoding UTF8|ConvertFrom-Json})
Check ($checks.Count -eq 26) 'Expected 26 finalist crosschecks'
$cross=New-Object System.Collections.Generic.List[object]
foreach($c in $checks){
 Check ($c.first_half_end_present -eq $true -and @($c.period_counts|Where-Object timeline_complete -ne $true).Count -eq 0) ('Incomplete ESPN timeline '+$c.id)
 $f=@($h|Where-Object id -eq $c.fotmob_id)[0]
 $firstHalfSame=($c.period_counts[0].first_half -eq [int]$f.home_corners_1h -and $c.period_counts[1].first_half -eq [int]$f.away_corners_1h)
 $cross.Add([pscustomobject]@{espn_id=$c.id;fotmob_id=$c.fotmob_id;date=$c.date;home=$c.home;away=$c.away;fotmob_1h=($f.home_corners_1h+'-'+$f.away_corners_1h);espn_1h=($c.period_counts[0].first_half.ToString()+'-'+$c.period_counts[1].first_half);first_half_agrees=$firstHalfSame;all_periods_agree=$c.all_periods_agree;source=$c.source})
}
Check (@($cross|Where-Object first_half_agrees -eq $true).Count -eq 25) 'Unexpected first half disagreement count'
$cross|Export-Csv (Join-Path $root 'provider_comparison.csv') -NoTypeInformation -Encoding UTF8
$selected=@($models|Where-Object id -eq '5836912')[0]
$aRows=@(Sample $selected 'home');$bRows=@(Sample $selected 'away')
$variants=New-Object System.Collections.Generic.List[object]
foreach($variant in @('baseline_fotmob','use_espn_derby_stoke_6_corners','exclude_derby_stoke_conflicting_observation','exclude_first_half_red_card_matches','exclude_all_red_card_matches')){
 $ar=@($aRows|ForEach-Object {$_|Select-Object *});$br=@($bRows|ForEach-Object {$_|Select-Object *})
 $bad=@()
 if($variant -eq 'use_espn_derby_stoke_6_corners'){foreach($r in $ar|Where-Object id -eq '5836877'){$r.home_corners_1h='3';$r.total_1h='6'}}
 elseif($variant -eq 'exclude_derby_stoke_conflicting_observation'){$bad=@('5836877')}
 elseif($variant -match '^exclude_.*red_card'){$bad=@($checks|Where-Object {($variant -eq 'exclude_all_red_card_matches' -and @($_.red_cards).Count -gt 0) -or ($variant -eq 'exclude_first_half_red_card_matches' -and @($_.red_cards|Where-Object period -eq 1).Count -gt 0)}|Select-Object -ExpandProperty fotmob_id)}
 $ar=@($ar|Where-Object {$_.id -notin $bad});$br=@($br|Where-Object {$_.id -notin $bad})
 $a=Stats $ar $selected.home_id;$b=Stats $br $selected.away_id;$av=Stats @($ar|Where-Object home_id -eq $selected.home_id) $selected.home_id;$bv=Stats @($br|Where-Object away_id -eq $selected.away_id) $selected.away_id
 $ls=(($a.cf+$a.ca)/$a.n+($b.cf+$b.ca)/$b.n)/2;$lv=(($av.cf+$av.ca)/$av.n+($bv.cf+$bv.ca)/$bv.n)/2;$lc=($ls+$lv)/2
 $variants.Add([pscustomobject]@{variant=$variant;home_n=$a.n;away_n=$b.n;home_venue_n=$av.n;away_venue_n=$bv.n;excluded_ids=$bad -join ';';lambda_season=$ls;lambda_venue=$lv;lambda_combined=$lc;p_raw=Cdf $lc;p_stress=Cdf (1.2*$lc)})
}
$variants|Export-Csv (Join-Path $root 'sensitivity.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{status='PASS';recorded_at_utc=$RecordedAtUtc;candidate_fixtures=20;eligible_models=19;excluded_fixture='Shakhtar-Sporting: 11 Shakhtar first-half histories missing';validated_histories=406;missing_first_half_histories=11;history_leagues=15;finalist_provider_checks=26;identical_first_half_totals=25;derby_stoke_first_half='FotMob 2-3, ESPN 3-3; Sportmonks/PlayerStats timeline supports 5 total; variants preserved';identical_all_periods=23;selected_id='5836912';p_raw=Num $selected.p_raw;p_stress=Num $selected.p_stress;calibrated=$false;future_results_present=$false}|ConvertTo-Json|Set-Content (Join-Path $root 'verification.json') -Encoding UTF8
$variants|ConvertTo-Json
