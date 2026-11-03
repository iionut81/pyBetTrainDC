$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$h=@(Import-Csv (Join-Path $root 'Data/historical_validated.csv') -Encoding UTF8)
$rejected=@(Import-Csv (Join-Path $root 'Data/historical_rejected.csv') -Encoding UTF8)
$global=Get-Content (Join-Path $root 'fixtures_global.json') -Raw -Encoding UTF8|ConvertFrom-Json
$ucl=Get-Content (Join-Path $root 'ESPN/uefa.champions.json') -Raw -Encoding UTF8|ConvertFrom-Json
$teamLeagues=@{'432'='tur.1';'134'='ger.1';'493'='ukr.1';'2250'='por.1';'1068'='esp.1';'132'='ger.1';'83'='esp.1';'362'='eng.1';
 '2980'='nor.1';'166'='fra.1';'142'='ned.1';'110'='ita.1';'4411'='aut.1';'521'='svk.1';'360'='eng.1';'104'='ita.1';'102'='esp.1';'160'='fra.1'}
$fixtures=New-Object System.Collections.Generic.List[object]
foreach($slug in @('eng.2','swe.1','uefa.champions')){
 $d=Get-Content (Join-Path $root ('ESPN/'+$slug+'.json')) -Raw -Encoding UTF8|ConvertFrom-Json
 foreach($f in $d.todays_events){
  $a=@($f.teams|Where-Object homeAway -eq 'home')[0];$b=@($f.teams|Where-Object homeAway -eq 'away')[0]
  $la=if($slug -eq 'uefa.champions'){$teamLeagues[[string]$a.id]}else{$slug}
  $lb=if($slug -eq 'uefa.champions'){$teamLeagues[[string]$b.id]}else{$slug}
  $aId=switch([string]$a.id){'493'{'9728'}'521'{'6019'}default{[string]$a.id}}
  $bId=switch([string]$b.id){'493'{'9728'}'521'{'6019'}default{[string]$b.id}}
  if(-not $la -or -not $lb){throw 'Missing domestic league crosswalk'}
  $fixtures.Add([pscustomobject]@{id=$f.id;competition=$slug;date=$f.date;home=$a.name;away=$b.name;home_history_id=$aId;away_history_id=$bId;
   home_competition_id=$a.id;away_competition_id=$b.id;home_league=$la;away_league=$lb;state=$f.state;time_valid=$f.time_valid;venue=$f.venue;source=$f.source;
   basis=if($slug -eq 'uefa.champions'){'domestic_league_projection_cross_competition'}else{'same_league_current_season'}})
 }
}
$cup=@($global.leagues|Where-Object {$_.name -eq 'DBU Pokalen' -and $_.country -eq 'DEN'})[0]
foreach($f in $cup.matches){
 $den=Get-Content (Join-Path $root 'ESPN/den.1.json') -Raw -Encoding UTF8|ConvertFrom-Json
 $teams=@($den.historical_events|ForEach-Object {$_.teams}|Sort-Object id -Unique)
 $a=@($teams|Where-Object name -match '^Lyngby');$b=@($teams|Where-Object name -match '^Randers')
 if($a.Count -ne 1 -or $b.Count -ne 1){throw 'Danish team crosswalk conflict'}
 $fixtures.Add([pscustomobject]@{id=$f.id;competition='den.cup';date=$f.date;home=$f.home;away=$f.away;home_history_id=$a[0].id;away_history_id=$b[0].id;
  home_competition_id=$a[0].id;away_competition_id=$b[0].id;home_league='den.1';away_league='den.1';state=if($f.started){'in'}else{'pre'};time_valid=($null -ne $f.date);
  venue='';source=('https://www.fotmob.com'+$f.pageUrl);basis='domestic_league_projection_to_cup'})
}
function Sample([string]$slug,[string]$id,[string]$venue='all'){
 @($h|Where-Object {$_.league -eq $slug -and (($venue -eq 'all' -and ($_.home_id -eq $id -or $_.away_id -eq $id)) -or
  ($venue -eq 'home' -and $_.home_id -eq $id) -or ($venue -eq 'away' -and $_.away_id -eq $id))})
}
function Stats($rows,[string]$id){
 $gf=0;$ga=0;$win=0;$late=0;$nil=0
 foreach($r in $rows){if($r.home_id -eq $id){$gf += [int]$r.home70;$ga += [int]$r.away70}else{$gf += [int]$r.away70;$ga += [int]$r.home70};if($r.success70 -eq 'True'){$win++}elseif($r.first_goal){$late++}else{$nil++}}
 [pscustomobject]@{n=@($rows).Count;gf=$gf;ga=$ga;win=$win;late=$late;nil=$nil}
}
$models=New-Object System.Collections.Generic.List[object]
foreach($f in $fixtures){
 $as=@(Sample $f.home_league $f.home_history_id);$bs=@(Sample $f.away_league $f.away_history_id)
 $avs=@(Sample $f.home_league $f.home_history_id 'home');$bvs=@(Sample $f.away_league $f.away_history_id 'away')
 $a=Stats $as $f.home_history_id;$b=Stats $bs $f.away_history_id;$av=Stats $avs $f.home_history_id;$bv=Stats $bvs $f.away_history_id
 $bad=@($rejected|Where-Object {($_.league -eq $f.home_league -and $_.name -match [regex]::Escape($f.home)) -or ($_.league -eq $f.away_league -and $_.name -match [regex]::Escape($f.away))})
 $eligible=($a.n -ge 5 -and $b.n -ge 5 -and $av.n -ge 3 -and $bv.n -ge 3 -and $bad.Count -eq 0 -and $f.state -eq 'pre' -and $f.time_valid)
 $ls=$null;$lv=$null;$lc=$null;$p=$null;$stress=$null;$fair=$null
 if($eligible){$ls=($a.gf/$a.n+$b.ga/$b.n+$b.gf/$b.n+$a.ga/$a.n)/2;$lv=($av.gf/$av.n+$bv.ga/$bv.n+$bv.gf/$bv.n+$av.ga/$av.n)/2;$lc=($ls+$lv)/2;$p=1-[math]::Exp(-$lc);$stress=1-[math]::Exp(-.8*$lc);$fair=1/$p}
 $unique=@(@($as)+@($bs)|Sort-Object league,id -Unique)
 $venueUnique=@(@($avs)+@($bvs)|Sort-Object league,id -Unique)
 $comp=@($h|Where-Object {$_.league -eq $f.competition -and ($_.home_id -in @($f.home_competition_id,$f.away_competition_id) -or $_.away_id -in @($f.home_competition_id,$f.away_competition_id))})
 $models.Add([pscustomobject]@{competition=$f.competition;id=$f.id;date=$f.date;home=$f.home;away=$f.away;home_history_id=$f.home_history_id;away_history_id=$f.away_history_id;
  home_league=$f.home_league;away_league=$f.away_league;eligible=$eligible;history_basis=$f.basis;venue=$f.venue;
  home_n=$a.n;home_gf70=$a.gf;home_ga70=$a.ga;home_success=$a.win;home_late=$a.late;home_0_0=$a.nil;
  away_n=$b.n;away_gf70=$b.gf;away_ga70=$b.ga;away_success=$b.win;away_late=$b.late;away_0_0=$b.nil;
  home_venue_n=$av.n;home_venue_gf70=$av.gf;home_venue_ga70=$av.ga;home_venue_success=$av.win;home_venue_late=$av.late;home_venue_0_0=$av.nil;
  away_venue_n=$bv.n;away_venue_gf70=$bv.gf;away_venue_ga70=$bv.ga;away_venue_success=$bv.win;away_venue_late=$bv.late;away_venue_0_0=$bv.nil;
  history_unique_n=$unique.Count;history_success=@($unique|Where-Object success70 -eq 'True').Count;
  venue_unique_n=$venueUnique.Count;venue_success=@($venueUnique|Where-Object success70 -eq 'True').Count;
  competition_history_n=$comp.Count;competition_history_success=@($comp|Where-Object success70 -eq 'True').Count;
  lambda_season=$ls;lambda_venue=$lv;lambda_combined=$lc;p_raw=$p;p_stress=$stress;fair_odds=$fair;rejected_team_matches=$bad.Count;source=$f.source})
}
$models|Sort-Object p_raw -Descending|Export-Csv (Join-Path $root 'all_models.csv') -NoTypeInformation -Encoding UTF8
$fixtures|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root 'candidate_fixtures.json') -Encoding UTF8
$models|Sort-Object p_raw -Descending|Select-Object home,away,competition,eligible,home_n,away_n,home_venue_n,away_venue_n,p_raw,p_stress,history_success,history_unique_n,venue_success,venue_unique_n|ConvertTo-Json
