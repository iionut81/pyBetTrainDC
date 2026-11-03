$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$goalRoot=Join-Path (Split-Path $root -Parent) 'Goal70_2026-11-03'
$g=Get-Content (Join-Path $goalRoot 'fixtures_global.json') -Raw -Encoding UTF8|ConvertFrom-Json
$models=@(Import-Csv (Join-Path $goalRoot 'all_models.csv') -Encoding UTF8)
$aliases=@{'Feyenoord'='Feyenoord Rotterdam';'Inter'='Internazionale';'LASK'='LASK Linz';'Roma'='AS Roma';'Sirius'='IK Sirius'}
$fixtures=New-Object System.Collections.Generic.List[object]
foreach($league in $g.leagues|Where-Object name -in @('Championship','Allsvenskan','Champions League','DBU Pokalen')){
 foreach($m in $league.matches){
  $ha=if($aliases.ContainsKey($m.home)){$aliases[$m.home]}else{$m.home}
  $aa=if($aliases.ContainsKey($m.away)){$aliases[$m.away]}else{$m.away}
  if($m.home_id -eq '8402'){$ha='Bodo/Glimt'}
  if($m.away_id -eq '9823'){$aa='Bayern Munich'}
  $match=@($models|Where-Object {$_.home -eq $ha -and $_.away -eq $aa})
  if($match.Count -ne 1 -or [DateTimeOffset]::Parse($match[0].date) -ne [DateTimeOffset]::Parse($m.date)){throw ('Identity crosswalk conflict '+$m.home+' - '+$m.away)}
  $x=$match[0]
  $fixtures.Add([pscustomobject]@{id=$m.id;goal70_espn_id=$x.id;competition=$x.competition;home=$m.home;away=$m.away;home_id=$m.home_id;away_id=$m.away_id;home_league=$x.home_league;away_league=$x.away_league;date=$m.date;started=$m.started;finished=$m.finished;cancelled=$m.cancelled;history_basis=$x.history_basis;source=('https://www.fotmob.com/api/data/matchDetails?matchId='+$m.id)})
 }
}
if($fixtures.Count -ne 20){throw 'Expected 20 fixtures'}
$fixtures|ConvertTo-Json -Depth 6|Set-Content (Join-Path $root 'candidate_fixtures.json') -Encoding UTF8
$leagueIds=@{'eng.2'=48;'eng.1'=47;'ita.1'=55;'esp.1'=87;'fra.1'=53;'por.1'=61;'tur.1'=71;'den.1'=46;'swe.1'=67;'ned.1'=57;'ger.1'=54;'aut.1'=38;'nor.1'=59;'ukr.1'=441;'svk.1'=176}
$defs=@(foreach($slug in $leagueIds.Keys|Sort-Object){
 $ids=@($fixtures|ForEach-Object {if($_.home_league -eq $slug){$_.home_id};if($_.away_league -eq $slug){$_.away_id}}|Sort-Object -Unique)
 if($ids.Count){[pscustomobject]@{league=$slug;parent_id=$leagueIds[$slug];season=if($slug -in @('swe.1','nor.1')){'2026'}else{'2026/2027'};team_ids=$ids}}
})
$defs|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root 'league_definitions.json') -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $goalRoot 'universe_coverage.csv') -Destination (Join-Path $root 'fixture_universe_reference.csv')
Copy-Item -LiteralPath (Join-Path $goalRoot 'exclusions.csv') -Destination (Join-Path $root 'fixture_exclusions_reference.csv')
'Prepared 20 fixture identities and '+$defs.Count+' domestic history leagues.'
