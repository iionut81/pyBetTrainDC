$ErrorActionPreference='Stop'
$url='https://www.fotmob.com/api/data/matches?date=20261103'
$d=(& curl.exe --compressed -L --max-time 25 --retry 1 -sS $url)|ConvertFrom-Json
if($LASTEXITCODE -ne 0){throw 'Current schedule fetch failed'}
$fixtures=@(Get-Content (Join-Path $PSScriptRoot 'candidate_fixtures.json') -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
$rows=New-Object System.Collections.Generic.List[object]
foreach($f in $fixtures){
 $m=@($d.leagues.matches|Where-Object {[string]$_.id -eq [string]$f.id})
 if($m.Count -ne 1){throw ('Current fixture missing or duplicate '+$f.id)};$m=$m[0]
 if([string]$m.home.id -ne [string]$f.home_id -or [string]$m.away.id -ne [string]$f.away_id -or [DateTimeOffset]::Parse($m.status.utcTime) -ne [DateTimeOffset]::Parse($f.date)){throw ('Identity or time changed '+$f.id)}
 if($m.status.started -or $m.status.finished -or $m.status.cancelled -or [DateTimeOffset]::Parse($m.status.utcTime) -le [DateTimeOffset]::UtcNow){throw ('No longer a live candidate '+$f.id)}
 $rows.Add([pscustomobject]@{id=[string]$m.id;home=$m.home.name;away=$m.away.name;date=$m.status.utcTime;started=$m.status.started;finished=$m.status.finished;cancelled=$m.status.cancelled;source=$url;verified_at_utc=[DateTime]::UtcNow.ToString('o')})
}
$rows|ConvertTo-Json -Depth 5|Set-Content (Join-Path $PSScriptRoot 'current_status.json') -Encoding UTF8
'Verified '+$rows.Count+' scheduled unstarted fixtures at '+[DateTime]::UtcNow.ToString('o')
