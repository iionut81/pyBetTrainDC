param([string]$TeamId,[string]$Label,[switch]$FullLeagueSeason,[int]$SeasonLeagueId=67)
$ErrorActionPreference='Stop'
$out=Join-Path $PSScriptRoot 'Finalists';[void](New-Item $out -ItemType Directory -Force)
function Fetch([string]$path){
 $raw=(& curl.exe --compressed -L --max-time 25 --retry 1 -sS ('https://www.fotmob.com/api/data/'+$path))
 if($LASTEXITCODE -ne 0){throw ('Fetch failed '+$path)};$raw|ConvertFrom-Json
}
$d=Fetch ('teams?id='+$TeamId)
$all=@($d.fixtures.allFixtures.fixtures|Where-Object {$_.status.finished -eq $true -and $_.status.cancelled -ne $true -and $_.status.awarded -ne $true -and $_.tournament.name -notmatch 'Friendl' -and [DateTimeOffset]::Parse($_.status.utcTime) -lt [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00')}|Sort-Object {$_.status.utcTime} -Descending)
$last=@($all|Select-Object -First 10)
$seasonStart=if($SeasonLeagueId -eq 67){[DateTimeOffset]::Parse('2026-01-01T00:00:00Z')}else{[DateTimeOffset]::Parse('2026-07-01T00:00:00Z')}
$selected=if($FullLeagueSeason){@(@($last)+@($all|Where-Object {$_.tournament.leagueId -eq $SeasonLeagueId -and [DateTimeOffset]::Parse($_.status.utcTime) -ge $seasonStart})|Sort-Object id -Unique)}else{$last}
$rows=New-Object System.Collections.Generic.List[object]
foreach($m in $selected){
 $cached=Join-Path $out ('match_'+$m.id+'.json')
 if(Test-Path -LiteralPath $cached){$rows.Add((Get-Content $cached -Raw -Encoding UTF8|ConvertFrom-Json));continue}
 $r=Fetch ('matchDetails?matchId='+$m.id)
 if([string]$r.general.matchId -ne [string]$m.id){throw 'Wrong match identity'}
 $goals=New-Object System.Collections.Generic.List[object]
 foreach($side in @('home','away')){
  $map=if($side -eq 'home'){$r.header.events.homeTeamGoals}else{$r.header.events.awayTeamGoals}
  foreach($p in $map.PSObject.Properties){foreach($g in @($p.Value)){
   if($g.type -ne 'Goal' -or $g.isPenaltyShootoutEvent){continue}
   $goals.Add([pscustomobject]@{minute=[int]$g.time;added=$g.overloadTime;side=$side;own_goal=[bool]$g.ownGoal;period=$g.shotmapEvent.period})
  }}
 }
 $cards=@($r.content.matchFacts.events.events|Where-Object {$_.type -match 'Card'}|ForEach-Object {[pscustomobject]@{type=$_.type;card=$_.card;minute=$_.time;added=$_.overloadTime;is_home=$_.isHome;name=$_.nameStr}})
 $regular=@($goals|Where-Object {$_.minute -le 90 -and $_.period -notmatch 'Extra|Shootout'})
 $first=@($regular|Sort-Object minute,added|Select-Object -First 1)
 $row=[pscustomobject]@{id=[string]$m.id;date=$m.status.utcTime;competition=$m.tournament.name;league_id=$m.tournament.leagueId;home=$m.home.name;away=$m.away.name;home_id=[string]$m.home.id;away_id=[string]$m.away.id;score=$m.status.scoreStr;status=$m.status.reason.short;goals=$goals;cards=$cards;first_goal=if($first.Count){[string]$first[0].minute+$(if($first[0].added -gt 0){'+'+[string]$first[0].added})}else{$null};success70=(@($regular|Where-Object minute -le 70).Count -gt 0);home_formation=$r.content.lineup.homeTeam.formation;away_formation=$r.content.lineup.awayTeam.formation;source=('https://www.fotmob.com'+$m.pageUrl);fetched_at_utc=[DateTime]::UtcNow.ToString('o')}
 if($m.status.reason.short -eq 'FT' -and $m.status.scoreStr -match '^(\d+)\s*-\s*(\d+)$'){
  if(@($goals|Where-Object side -eq 'home').Count -ne [int]$Matches[1] -or @($goals|Where-Object side -eq 'away').Count -ne [int]$Matches[2]){throw ('Score reconciliation failed '+$m.id)}
 }
 $row|ConvertTo-Json -Depth 8|Set-Content $cached -Encoding UTF8;$rows.Add($row)
}
$lastIds=@($last|ForEach-Object {[string]$_.id})
$rows|Where-Object {$_.id -in $lastIds}|Sort-Object date -Descending|ConvertTo-Json -Depth 8|Set-Content (Join-Path $out ($Label+'_last10.json')) -Encoding UTF8
if($FullLeagueSeason){$rows|Where-Object {$_.league_id -eq $SeasonLeagueId -and [DateTimeOffset]::Parse($_.date) -ge $seasonStart}|Sort-Object date|ConvertTo-Json -Depth 8|Set-Content (Join-Path $out ($Label+'_league.json')) -Encoding UTF8}
Write-Output ($Label+': verified '+$rows.Count+' official matches; last10='+$last.Count)
