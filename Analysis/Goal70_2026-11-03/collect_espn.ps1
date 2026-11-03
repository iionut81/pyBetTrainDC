param([int]$Shard=0,[int]$ShardCount=4)
$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot 'ESPN';[void](New-Item $root -ItemType Directory -Force)
$definitions=@(
 @('eng.1','England',1),@('ita.1','Italy',2),@('esp.1','Spain',3),@('ger.1','Germany',4),@('fra.1','France',5),
 @('por.1','Portugal',6),@('ned.1','Netherlands',7),@('bel.1','Belgium',8),@('tur.1','Turkey',9),@('cze.1','Czechia',10),
 @('gre.1','Greece',11),@('pol.1','Poland',12),@('den.1','Denmark',13),@('nor.1','Norway',14),@('cyp.1','Cyprus',15),
 @('sui.1','Switzerland',16),@('aut.1','Austria',17),@('sco.1','Scotland',18),@('swe.1','Sweden',19),@('cro.1','Croatia',20),
 @('rou.1','Romania',0),@('srb.1','Serbia',0),@('eng.2','England 2',0),@('ger.2','Germany 2',0),@('esp.2','Spain 2',0),
 @('ita.2','Italy 2',0),@('fra.2','France 2',0),@('bel.2','Belgium 2',0),@('tur.2','Turkey 2',0),
 @('ukr.1','Ukraine: UCL participant history',0),@('svk.1','Slovakia: UCL participant history',0),
 @('uefa.champions','Champions League',0)
)
$historyLeagues=@('eng.1','ita.1','esp.1','ger.1','fra.1','por.1','ned.1','tur.1','den.1','nor.1','aut.1','swe.1','eng.2','ukr.1','svk.1','uefa.champions')
function Fetch([string]$url){
 $r=@(& curl.exe --compressed -L --max-time 22 -sS -w '\nHTTP_STATUS:%{http_code}' $url 2>$null)
 $raw=$r -join "`n";$ix=$raw.LastIndexOf('HTTP_STATUS:')
 if($LASTEXITCODE -ne 0 -or $ix -lt 0){return @{ok=$false;error='transport'}}
 $code=$raw.Substring($ix+12).Trim();if($code -ne '200'){return @{ok=$false;error=('http_'+$code)}}
 try{$d=$raw.Substring(0,$ix).Trim()|ConvertFrom-Json}catch{return @{ok=$false;error='invalid_json'}}
 @{ok=$true;data=$d;source=$url}
}
function Event($e,[string]$slug){
 $c=$e.competitions[0]
 [pscustomobject]@{id=[string]$e.id;league=$slug;date=$e.date;season_year=$e.season.year;name=$e.name;state=$c.status.type.state;status=$c.status.type.name;
  completed=$c.status.type.completed;time_valid=$c.timeValid;venue=$c.venue.fullName;
  teams=@($c.competitors|ForEach-Object {[pscustomobject]@{id=[string]$_.id;name=$_.team.displayName;homeAway=$_.homeAway;score=$_.score}});
  goals=@($c.details|Where-Object {$_.scoringPlay -eq $true -and $_.shootout -ne $true}|ForEach-Object {[pscustomobject]@{minute=$_.clock.displayValue;team_id=[string]$_.team.id;own_goal=[bool]$_.ownGoal;type_id=$_.type.id;score_value=$_.scoreValue}});
  source=('https://www.espn.com/soccer/match/_/gameId/'+$e.id)}
}
for($i=0;$i -lt $definitions.Count;$i++){
 if(($i%$ShardCount) -ne $Shard){continue};$def=$definitions[$i];$slug=$def[0]
 $file=Join-Path $root ($slug+'.json');if(Test-Path -LiteralPath $file){continue}
 $today=Fetch ('https://site.api.espn.com/apis/site/v2/sports/soccer/'+$slug+'/scoreboard?dates=20261103&limit=1000')
 $out=[ordered]@{league=$slug;country=$def[1];uefa_2026_rank=$def[2];as_of='2026-11-03';fetched_at_utc=[DateTime]::UtcNow.ToString('o');
  today_ok=$false;history_ok=$false;today_error=$null;history_error=$null;todays_events=@();historical_events=@()}
 if($today.ok -and $today.data.leagues[0].slug -eq $slug){$out.today_ok=$true;$out.today_source=$today.source;$out.todays_events=@($today.data.events|ForEach-Object {Event $_ $slug})}else{$out.today_error=$today.error}
 if($slug -in $historyLeagues){
  $hist=Fetch ('https://site.api.espn.com/apis/site/v2/sports/soccer/'+$slug+'/scoreboard?dates=2026&limit=1000')
  if($hist.ok -and $hist.data.leagues[0].slug -eq $slug){
   $out.history_ok=$true;$out.history_source=$hist.source
   $out.historical_events=@($hist.data.events|Where-Object {$_.season.year -eq 2026 -and [DateTimeOffset]::Parse($_.date) -lt [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00') -and $_.competitions[0].status.type.completed -eq $true}|ForEach-Object {Event $_ $slug})
  }else{$out.history_error=$hist.error}
 }
 $out|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $file -Encoding UTF8
 Write-Output ($slug+': daily='+@($out.todays_events).Count+' history='+@($out.historical_events).Count+' errors='+$out.today_error+'/'+$out.history_error)
}
