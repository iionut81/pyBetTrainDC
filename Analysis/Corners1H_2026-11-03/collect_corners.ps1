param([int]$Shard=0,[int]$ShardCount=4)
$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot 'FotMob';[void](New-Item $root -ItemType Directory -Force)
$oldRoot=Join-Path (Split-Path $PSScriptRoot -Parent) 'Corners1H_2026-11-02/FotMob'
$defs=@(Get-Content (Join-Path $PSScriptRoot 'league_definitions.json') -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
function Fetch([string]$path){
 $url='https://www.fotmob.com/api/data/'+$path
 $r=@(& curl.exe --compressed -L --max-time 22 --retry 1 -sS -w '\nHTTP_STATUS:%{http_code}' $url 2>$null)
 $raw=$r -join "`n";$ix=$raw.LastIndexOf('HTTP_STATUS:')
 if($LASTEXITCODE -ne 0 -or $ix -lt 0){return @{ok=$false;error='transport_error';source=$url}}
 $code=$raw.Substring($ix+12).Trim();if($code -ne '200'){return @{ok=$false;error=('http_'+$code);source=$url}}
 try{$data=$raw.Substring(0,$ix).Trim()|ConvertFrom-Json}catch{return @{ok=$false;error='invalid_json';source=$url}}
 @{ok=$true;data=$data;source=$url}
}
function Fixture($m,[string]$slug){[pscustomobject]@{id=[string]$m.id;league=$slug;date=$m.status.utcTime;home_id=[string]$m.home.id;home=$m.home.name;away_id=[string]$m.away.id;away=$m.away.name;finished=$m.status.finished;started=$m.status.started;cancelled=$m.status.cancelled;awarded=$m.status.awarded;score=$m.status.scoreStr;source=('https://www.fotmob.com'+$m.pageUrl)}}
function Corner($d,[string]$period){
 $c=@($d.content.stats.Periods.$period.stats|ForEach-Object {$_.stats}|Where-Object key -eq 'corners')
 if($c.Count -ne 1 -or @($c[0].stats).Count -ne 2){return $null}
 if($null -eq $c[0].stats[0] -or $null -eq $c[0].stats[1] -or [string]$c[0].stats[0] -notmatch '^\d+$' -or [string]$c[0].stats[1] -notmatch '^\d+$'){return $null}
 [pscustomobject]@{home=[int]$c[0].stats[0];away=[int]$c[0].stats[1];key='corners';period=$period}
}
for($i=0;$i -lt $defs.Count;$i++){
 if(($i%$ShardCount) -ne $Shard){continue}
 $def=$defs[$i];$slug=$def.league
 $schedule=Fetch ('leagues?id='+$def.parent_id+'&ccode3=ROU&season='+[Uri]::EscapeDataString($def.season))
 if(-not $schedule.ok){[pscustomobject]@{league=$slug;ok=$false;error=$schedule.error;source=$schedule.source}|ConvertTo-Json|Set-Content (Join-Path $root ($slug+'.json')) -Encoding UTF8;Write-Output ($slug+' schedule failed '+$schedule.error);continue}
 if($schedule.data.details.selectedSeason -ne $def.season){throw ('Wrong season '+$slug)}
 $all=@($schedule.data.fixtures.allMatches|ForEach-Object {Fixture $_ $slug})
 $past=@($all|Where-Object {$_.finished -eq $true -and $_.cancelled -ne $true -and $_.awarded -ne $true -and [DateTimeOffset]::Parse($_.date) -lt [DateTimeOffset]::Parse('2026-11-03T00:00:00+02:00') -and ($_.home_id -in $def.team_ids -or $_.away_id -in $def.team_ids)})
 foreach($team in $def.team_ids){if(@($past|Where-Object {$_.home_id -eq $team -or $_.away_id -eq $team}).Count -eq 0){throw ('Team absent from selected league '+$slug+' '+$team)}}
 [pscustomobject]@{league=$slug;ok=$true;name=$schedule.data.details.name;season=$def.season;parent_id=$def.parent_id;team_ids=$def.team_ids;expected_history=$past;source=$schedule.source;collected_at_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Depth 7|Set-Content (Join-Path $root ($slug+'.json')) -Encoding UTF8
 $count=0;$reused=0
 foreach($m in $past){
  $dest=Join-Path $root ('match_'+$slug+'_'+$m.id+'.json')
  if(Test-Path -LiteralPath $dest){continue}
  $old=Join-Path $oldRoot ('match_'+$slug+'_'+$m.id+'.json')
  if(Test-Path -LiteralPath $old){$cached=Get-Content $old -Raw -Encoding UTF8|ConvertFrom-Json;if($cached.ok -and $cached.fixture.id -eq $m.id -and $cached.fixture.home_id -eq $m.home_id -and $cached.fixture.away_id -eq $m.away_id){Copy-Item -LiteralPath $old -Destination $dest;$reused++;continue}}
  $r=Fetch ('matchDetails?matchId='+$m.id)
  $record=[ordered]@{fixture=$m;ok=$false;error=$null;source=$r.source;collected_at_utc=[DateTime]::UtcNow.ToString('o');period_definition='content.stats.Periods.FirstHalf key corners, including first-half added time';first_half=$null;second_half=$null;full_time=$null;reconciled=$false}
  if(-not $r.ok){$record.error=$r.error}else{
   $d=$r.data
   if([string]$d.general.matchId -ne $m.id -or [string]$d.general.homeTeam.id -ne $m.home_id -or [string]$d.general.awayTeam.id -ne $m.away_id){throw ('Identity mismatch '+$m.id)}
   $record.first_half=Corner $d 'FirstHalf';$record.second_half=Corner $d 'SecondHalf';$record.full_time=Corner $d 'All'
   if($null -eq $record.first_half){$record.error='first_half_corners_missing'}elseif($null -eq $record.second_half -or $null -eq $record.full_time){$record.error='cannot_reconcile_periods'}elseif($record.first_half.home+$record.second_half.home -ne $record.full_time.home -or $record.first_half.away+$record.second_half.away -ne $record.full_time.away){$record.error='period_sum_mismatch'}else{$record.ok=$true;$record.reconciled=$true}
  }
  $record|ConvertTo-Json -Depth 8|Set-Content $dest -Encoding UTF8
  $count++;if($count%25 -eq 0){Write-Output ($slug+': fetched '+$count+'/'+$past.Count+', reused '+$reused)}
 }
 Write-Output ($slug+' complete: '+$past.Count+' distinct expected, '+$count+' fetched, '+$reused+' reused')
}
