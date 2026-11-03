$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$analysis=Split-Path $root -Parent
[void](New-Item (Join-Path $root 'FotMob') -ItemType Directory -Force)
[void](New-Item (Join-Path $root 'ESPN') -ItemType Directory -Force)
$models=@(Import-Csv (Join-Path $analysis 'Corners1H_2026-11-02/all_models.csv') -Encoding UTF8)
function Fetch([string]$url){
 $r=@(& curl.exe --compressed -L --max-time 22 -sS -w '\nHTTP_STATUS:%{http_code}' $url 2>$null)
 $t=$r -join "`n";$ix=$t.LastIndexOf('HTTP_STATUS:')
 if($LASTEXITCODE -ne 0 -or $ix -lt 0){return @{ok=$false;error='transport';url=$url}}
 $code=$t.Substring($ix+12).Trim();if($code -ne '200'){return @{ok=$false;error=('http_'+$code);url=$url}}
 try{$d=$t.Substring(0,$ix).Trim()|ConvertFrom-Json}catch{return @{ok=$false;error='invalid_json';url=$url}}
 @{ok=$true;data=$d;url=$url}
}
foreach($m in $models){
 $file=Join-Path $root ('FotMob/'+$m.id+'.json')
 if(Test-Path -LiteralPath $file){continue}
 $r=Fetch ('https://www.fotmob.com/api/data/matchDetails?matchId='+$m.id)
 if(-not $r.ok){$r|ConvertTo-Json|Set-Content $file -Encoding UTF8;continue}
 $d=$r.data
 if([string]$d.general.matchId -ne $m.id -or [string]$d.general.homeTeam.id -ne $m.home_id -or [string]$d.general.awayTeam.id -ne $m.away_id){throw ('Identity mismatch '+$m.id)}
 $goals=New-Object System.Collections.Generic.List[object]
 foreach($side in @('home','away')){
  $map=if($side -eq 'home'){$d.header.events.homeTeamGoals}else{$d.header.events.awayTeamGoals}
  foreach($prop in $map.PSObject.Properties){foreach($g in @($prop.Value)){
   if($g.type -ne 'Goal'){continue}
   $goals.Add([pscustomobject]@{side=$side;minute=$g.time;added_time=$g.overloadTime;time_str=$g.timeStr;event_id=$g.eventId;
    home_score=$g.homeScore;away_score=$g.awayScore;goal_description=$g.goalDescription;period=$g.shotmapEvent.period;own_goal=$g.shotmapEvent.isOwnGoal})
  }}
 }
 $stats=New-Object System.Collections.Generic.List[object]
 foreach($period in @('All','FirstHalf','SecondHalf')){
  foreach($s in @($d.content.stats.Periods.$period.stats|ForEach-Object {$_.stats}|Where-Object {$_.key -in @('expected_goals','total_shots','ShotsOnTarget','corners','BallPossesion','big_chance','blocked_shots','touches_opp_box')})){
   $stats.Add([pscustomobject]@{period=$period;key=$s.key;title=$s.title;values=$s.stats})
  }
 }
 [pscustomobject]@{ok=$true;league=$m.league;id=$m.id;source=$r.url;match_source=$m.source;fetched_at_utc=[DateTime]::UtcNow.ToString('o');
  general=$d.general|Select-Object matchId,leagueName,parentLeagueId,matchTimeUTC,started,finished,homeTeam,awayTeam;
  status=$d.header.status;teams=@($d.header.teams|Select-Object id,name,score);goals=$goals;stats=$stats;
  cards=@($d.content.matchFacts.events.events|Where-Object {$_.type -eq 'Card'}|Select-Object type,time,timeStr,isHome,card,cardType,overloadTime);
  formations=$d.content.lineup|Select-Object homeFormation,awayFormation}|
 ConvertTo-Json -Depth 10|Set-Content -LiteralPath $file -Encoding UTF8
 Write-Output ($m.home+' - '+$m.away+' collected')
}
$slugs=@($models.league|Where-Object {$_ -notin @('pol.1','rou.1','srb.1','tur.2')}|Select-Object -Unique)
foreach($slug in $slugs){
 $file=Join-Path $root ('ESPN/'+$slug+'.json')
 if(Test-Path -LiteralPath $file){continue}
 $r=Fetch ('https://site.api.espn.com/apis/site/v2/sports/soccer/'+$slug+'/scoreboard?dates=20261102&limit=1000')
 if(-not $r.ok){$r|ConvertTo-Json|Set-Content $file -Encoding UTF8;continue}
 $events=@($r.data.events|ForEach-Object {
  $e=$_;$c=$e.competitions[0]
  [pscustomobject]@{id=[string]$e.id;date=$e.date;state=$c.status.type.state;completed=$c.status.type.completed;status=$c.status.type.name;
   teams=@($c.competitors|ForEach-Object {[pscustomobject]@{id=[string]$_.id;name=$_.team.displayName;homeAway=$_.homeAway;score=$_.score}});
   goals=@($c.details|Where-Object {$_.scoringPlay -eq $true -and $_.shootout -ne $true}|ForEach-Object {[pscustomobject]@{minute=$_.clock.displayValue;team_id=[string]$_.team.id;own_goal=$_.ownGoal}});
   source=('https://www.espn.com/soccer/match/_/gameId/'+$e.id)}
 })
 [pscustomobject]@{ok=$true;league=$slug;source=$r.url;fetched_at_utc=[DateTime]::UtcNow.ToString('o');events=$events}|ConvertTo-Json -Depth 9|Set-Content $file -Encoding UTF8
 Write-Output ($slug+' ESPN '+$events.Count)
}
