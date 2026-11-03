$ErrorActionPreference='Stop'
$rows=New-Object System.Collections.Generic.List[object]
foreach($slug in @('eng.2','swe.1','uefa.champions')){
 $url='https://site.api.espn.com/apis/site/v2/sports/soccer/'+$slug+'/scoreboard?dates=20261103&limit=1000'
 $d=(& curl.exe --compressed -L --max-time 25 -sS $url)|ConvertFrom-Json
 if($LASTEXITCODE -ne 0){throw 'Status refresh failed'}
 foreach($e in $d.events){$c=$e.competitions[0];$rows.Add([pscustomobject]@{id=[string]$e.id;competition=$slug;date=$e.date;state=$e.status.type.state;status=$e.status.type.name;home=($c.competitors|Where-Object homeAway -eq 'home').team.displayName;away=($c.competitors|Where-Object homeAway -eq 'away').team.displayName;source=$url;checked_at_utc=[DateTime]::UtcNow.ToString('o')})}
}
$url='https://www.fotmob.com/api/data/matches?date=20261103'
$d=(& curl.exe --compressed -L --max-time 25 -sS $url)|ConvertFrom-Json
if($LASTEXITCODE -ne 0){throw 'Global status refresh failed'}
$cup=$d.leagues|Where-Object {$_.ccode -eq 'DEN' -and $_.name -eq 'DBU Pokalen'}
if(-not $cup){$cup=$d.leagues|Where-Object name -eq 'DBU Pokalen'}
foreach($m in $cup.matches){$rows.Add([pscustomobject]@{id=[string]$m.id;competition='den.cup';date=$m.status.utcTime;state=if($m.status.started){'in'}else{'pre'};status=if($m.status.cancelled){'cancelled'}else{'scheduled'};home=$m.home.name;away=$m.away.name;source=$url;checked_at_utc=[DateTime]::UtcNow.ToString('o')})}
$models=@(Import-Csv (Join-Path $PSScriptRoot 'all_models.csv'))
if($rows.Count -ne $models.Count){throw ('Status refresh coverage: '+$rows.Count+' vs '+$models.Count)}
foreach($x in $models){$r=@($rows|Where-Object {$_.id -eq $x.id -and $_.competition -eq $x.competition});if($r.Count -ne 1 -or $r[0].state -ne 'pre' -or $r[0].status -eq 'cancelled' -or [DateTimeOffset]::Parse($r[0].date) -le [DateTimeOffset]::UtcNow -or [DateTimeOffset]::Parse($r[0].date) -ne [DateTimeOffset]::Parse($x.date)){throw ('Status conflict '+$x.id)}}
$rows|ConvertTo-Json -Depth 5|Set-Content (Join-Path $PSScriptRoot 'live_status.json') -Encoding UTF8
Write-Output ('Verified all '+$rows.Count+' fixtures are scheduled and unstarted at '+[DateTime]::UtcNow.ToString('o'))
