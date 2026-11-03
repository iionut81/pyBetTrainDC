param([string]$League)
$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot 'FotMob';[void](New-Item $root -ItemType Directory -Force)
$definition=switch($League){'ukr.1'{@{id=441;team='9728';name='Ukraine'}}'svk.1'{@{id=176;team='6019';name='Slovakia'}}default{throw 'Unknown fallback league'}}
function Fetch([string]$path){$url='https://www.fotmob.com/api/data/'+$path;$raw=(& curl.exe --compressed -L --max-time 22 -sS $url);if($LASTEXITCODE -ne 0){throw ('Request failed '+$path)};@{data=($raw|ConvertFrom-Json);source=$url}}
$r=Fetch ('leagues?id='+$definition.id+'&ccode3=ROU&season=2026%2F2027')
if($r.data.details.selectedSeason -ne '2026/2027'){throw 'Wrong season'}
$fixtures=@($r.data.fixtures.allMatches|Where-Object {$_.status.finished -eq $true -and $_.status.cancelled -ne $true -and $_.status.awarded -ne $true -and
 [DateTimeOffset]::Parse($_.status.utcTime) -lt [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00') -and
 ([string]$_.home.id -eq $definition.team -or [string]$_.away.id -eq $definition.team)})
$events=New-Object System.Collections.Generic.List[object]
foreach($m in $fixtures){
 if($m.status.scoreStr -notmatch '^(\d+)\s*-\s*(\d+)$'){throw ('Missing FT fixture score '+$m.id)}
 $homeScore=[string]$Matches[1];$awayScore=[string]$Matches[2]
 $cached=Join-Path $root ('match_'+$m.id+'.json')
 if(Test-Path -LiteralPath $cached){
  $event=Get-Content $cached -Raw -Encoding UTF8|ConvertFrom-Json
  ($event.teams|Where-Object homeAway -eq 'home').score=$homeScore
  ($event.teams|Where-Object homeAway -eq 'away').score=$awayScore
  $event|ConvertTo-Json -Depth 8|Set-Content $cached -Encoding UTF8
  $events.Add($event);continue
 }
 $res=Fetch ('matchDetails?matchId='+$m.id);$d=$res.data
 if([string]$d.general.matchId -ne [string]$m.id){throw 'Wrong match identity'}
 $goals=New-Object System.Collections.Generic.List[object]
 foreach($side in @('home','away')){
  $goalMap=if($side -eq 'home'){$d.header.events.homeTeamGoals}else{$d.header.events.awayTeamGoals}
  foreach($p in $goalMap.PSObject.Properties){foreach($g in @($p.Value)){
   if($g.type -ne 'Goal'){continue};$minute=[string]$g.time+"'";if($g.overloadTime -gt 0){$minute+='+'+[string]$g.overloadTime+"'"}
   $goals.Add([pscustomobject]@{minute=$minute;team_id=if($side -eq 'home'){[string]$m.home.id}else{[string]$m.away.id};own_goal=[bool]$g.shotmapEvent.isOwnGoal})
  }}
 }
 $event=[pscustomobject]@{league=$League;id=[string]$m.id;date=$m.status.utcTime;name=($m.away.name+' at '+$m.home.name);completed=$true;state='post';status='STATUS_FULL_TIME';time_valid=$true;
  teams=@([pscustomobject]@{id=[string]$m.home.id;name=$m.home.name;homeAway='home';score=$homeScore},[pscustomobject]@{id=[string]$m.away.id;name=$m.away.name;homeAway='away';score=$awayScore});
  goals=$goals;source=('https://www.fotmob.com'+$m.pageUrl);api_source=$res.source;fetched_at_utc=[DateTime]::UtcNow.ToString('o')}
 $event|ConvertTo-Json -Depth 8|Set-Content $cached -Encoding UTF8;$events.Add($event)
}
[pscustomobject]@{league=$League;country=$definition.name;history_ok=$true;historical_events=$events;source=$r.source;team_id=$definition.team;season='2026/2027';fetched_at_utc=[DateTime]::UtcNow.ToString('o')}|
 ConvertTo-Json -Depth 10|Set-Content (Join-Path $root ($League+'.json')) -Encoding UTF8
Write-Output ($League+' '+$events.Count+' matches')
