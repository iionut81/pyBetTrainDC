$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot 'FotMob'
$history=New-Object System.Collections.Generic.List[object]
$rejected=New-Object System.Collections.Generic.List[object]
foreach($file in Get-ChildItem $root -Filter 'match_*.json'){
 $d=Get-Content $file.FullName -Raw -Encoding UTF8|ConvertFrom-Json;$m=$d.fixture
 if(-not $d.ok){$rejected.Add([pscustomobject]@{league=$m.league;id=$m.id;date=$m.date;home=$m.home;away=$m.away;reason=$d.error;source=$d.source});continue}
 if(-not $d.reconciled -or $d.first_half.period -ne 'FirstHalf' -or $d.first_half.key -ne 'corners'){throw ('Invalid period '+$m.id)}
 foreach($side in @('home','away')){
  foreach($period in @('first_half','second_half','full_time')){if([string]$d.$period.$side -notmatch '^\d+$'){throw ('Invalid corner count '+$m.id)}}
  if($d.first_half.$side+$d.second_half.$side -ne $d.full_time.$side){throw ('Period sum mismatch '+$m.id)}
 }
 if([DateTimeOffset]::Parse($m.date) -ge [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00')){throw ('Future history '+$m.id)}
 $total=$d.first_half.home+$d.first_half.away
 $history.Add([pscustomobject]@{league=$m.league;id=$m.id;date=$m.date;home_id=$m.home_id;home=$m.home;away_id=$m.away_id;away=$m.away;home_corners_1h=$d.first_half.home;away_corners_1h=$d.first_half.away;total_1h=$total;under_7_5=($total -le 7);exactly_7=($total -eq 7);eight_plus=($total -ge 8);home_corners_2h=$d.second_half.home;away_corners_2h=$d.second_half.away;home_corners_ft=$d.full_time.home;away_corners_ft=$d.full_time.away;source=$m.source;api_source=$d.source;period='FirstHalf';collected_at_utc=$d.collected_at_utc})
}
if(@($history|Group-Object league,id|Where-Object Count -gt 1).Count){throw 'Duplicate historical match'}
function Cdf7([double]$lambda){$term=[math]::Exp(-$lambda);$sum=$term;for($k=1;$k -le 7;$k++){$term*=$lambda/$k;$sum+=$term};$sum}
function Sample([string]$league,[string]$team,[string]$venue='all'){
 @($history|Where-Object {$_.league -eq $league -and (($venue -eq 'all' -and ($_.home_id -eq $team -or $_.away_id -eq $team)) -or ($venue -eq 'home' -and $_.home_id -eq $team) -or ($venue -eq 'away' -and $_.away_id -eq $team))})
}
function Stats($rows,[string]$team){
 $cf=0;$ca=0;$under=0;$seven=0;$eight=0;$variance=$null
 foreach($r in $rows){if($r.home_id -eq $team){$cf += $r.home_corners_1h;$ca += $r.away_corners_1h}else{$cf += $r.away_corners_1h;$ca += $r.home_corners_1h};if($r.under_7_5){$under++};if($r.exactly_7){$seven++};if($r.eight_plus){$eight++}}
 $n=@($rows).Count;$mean=if($n){($cf+$ca)/$n}else{$null}
 if($n -gt 1){$ss=0;foreach($r in $rows){$ss += [math]::Pow($r.total_1h-$mean,2)};$variance=$ss/($n-1)}
 [pscustomobject]@{n=$n;cf=$cf;ca=$ca;under=$under;seven=$seven;eight=$eight;mean=$mean;variance=$variance}
}
$coverage=New-Object System.Collections.Generic.List[object]
$records=@{}
foreach($file in Get-ChildItem $root -Filter '*.json'|Where-Object Name -match '^[a-z]{3}\.[12]\.json$'){
 $d=Get-Content $file.FullName -Raw -Encoding UTF8|ConvertFrom-Json;$records[$d.league]=$d
 $valid=@($history|Where-Object league -eq $d.league).Count;$bad=@($rejected|Where-Object league -eq $d.league).Count
 $coverage.Add([pscustomobject]@{league=$d.league;season=$d.season;league_ok=$d.ok;expected=@($d.expected_history).Count;validated=$valid;rejected=$bad;not_fetched=@($d.expected_history).Count-$valid-$bad;source=$d.source})
}
$fixtures=@(Get-Content (Join-Path $PSScriptRoot 'candidate_fixtures.json') -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
$models=New-Object System.Collections.Generic.List[object]
foreach($f in $fixtures){
 $ar=@(Sample $f.home_league $f.home_id);$br=@(Sample $f.away_league $f.away_id)
 $avr=@(Sample $f.home_league $f.home_id 'home');$bvr=@(Sample $f.away_league $f.away_id 'away')
 $a=Stats $ar $f.home_id;$b=Stats $br $f.away_id;$av=Stats $avr $f.home_id;$bv=Stats $bvr $f.away_id
 $ad=$records[$f.home_league];$bd=$records[$f.away_league]
 $ea=@($ad.expected_history|Where-Object {$_.home_id -eq $f.home_id -or $_.away_id -eq $f.home_id}).Count
 $eb=@($bd.expected_history|Where-Object {$_.home_id -eq $f.away_id -or $_.away_id -eq $f.away_id}).Count
 $eligible=($ad.ok -eq $true -and $bd.ok -eq $true -and $a.n -eq $ea -and $b.n -eq $eb -and $a.n -ge 5 -and $b.n -ge 5 -and $av.n -ge 3 -and $bv.n -ge 3 -and $f.started -ne $true -and $f.finished -ne $true -and $f.cancelled -ne $true)
 $ls=$null;$lv=$null;$lc=$null;$p=$null;$stress=$null;$fair=$null
 if($eligible){$ls=($a.cf/$a.n+$b.ca/$b.n+$b.cf/$b.n+$a.ca/$a.n)/2;$lv=($av.cf/$av.n+$bv.ca/$bv.n+$bv.cf/$bv.n+$av.ca/$av.n)/2;$lc=($ls+$lv)/2;$p=Cdf7 $lc;$stress=Cdf7 (1.2*$lc);$fair=1/$p}
 $sample=@(@($ar)+@($br)|Sort-Object league,id -Unique);$venue=@(@($avr)+@($bvr)|Sort-Object league,id -Unique)
 $models.Add([pscustomobject]@{competition=$f.competition;id=$f.id;goal70_espn_id=$f.goal70_espn_id;date=$f.date;home=$f.home;home_id=$f.home_id;away=$f.away;away_id=$f.away_id;home_league=$f.home_league;away_league=$f.away_league;history_basis=$f.history_basis;eligible=$eligible;reason=if($eligible){'complete_current_domestic_season'}else{'missing_or_insufficient_history_or_status'};home_expected=$ea;away_expected=$eb;
 home_n=$a.n;home_cf=$a.cf;home_ca=$a.ca;home_under=$a.under;home_7=$a.seven;home_8plus=$a.eight;home_mean=$a.mean;home_variance=$a.variance;
 away_n=$b.n;away_cf=$b.cf;away_ca=$b.ca;away_under=$b.under;away_7=$b.seven;away_8plus=$b.eight;away_mean=$b.mean;away_variance=$b.variance;
 home_venue_n=$av.n;home_venue_cf=$av.cf;home_venue_ca=$av.ca;home_venue_under=$av.under;home_venue_7=$av.seven;home_venue_8plus=$av.eight;home_venue_mean=$av.mean;home_venue_variance=$av.variance;
 away_venue_n=$bv.n;away_venue_cf=$bv.cf;away_venue_ca=$bv.ca;away_venue_under=$bv.under;away_venue_7=$bv.seven;away_venue_8plus=$bv.eight;away_venue_mean=$bv.mean;away_venue_variance=$bv.variance;
 unique_n=$sample.Count;unique_under=@($sample|Where-Object under_7_5).Count;unique_7=@($sample|Where-Object exactly_7).Count;unique_8plus=@($sample|Where-Object eight_plus).Count;
 venue_unique_n=$venue.Count;venue_under=@($venue|Where-Object under_7_5).Count;venue_7=@($venue|Where-Object exactly_7).Count;venue_8plus=@($venue|Where-Object eight_plus).Count;
 lambda_season=$ls;lambda_venue=$lv;lambda_combined=$lc;p_raw=$p;p_stress=$stress;fair_odds=$fair;source=$f.source})
}
$history|Export-Csv (Join-Path $PSScriptRoot 'historical_corners_1h.csv') -NoTypeInformation -Encoding UTF8
$rejected|Export-Csv (Join-Path $PSScriptRoot 'rejected_corners.csv') -NoTypeInformation -Encoding UTF8
$coverage|Export-Csv (Join-Path $PSScriptRoot 'coverage.csv') -NoTypeInformation -Encoding UTF8
$models|Sort-Object p_raw -Descending|Export-Csv (Join-Path $PSScriptRoot 'all_models.csv') -NoTypeInformation -Encoding UTF8
$models|Sort-Object p_raw -Descending|Select-Object home,away,competition,eligible,home_n,away_n,home_venue_n,away_venue_n,unique_under,unique_n,unique_8plus,unique_7,venue_under,venue_unique_n,venue_7,lambda_combined,p_raw,p_stress|ConvertTo-Json
'Historical valid '+$history.Count+'; rejected '+$rejected.Count+'; candidates '+$models.Count
