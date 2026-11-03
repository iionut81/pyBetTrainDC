param([int]$Shard=0,[int]$ShardCount=4)
$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot 'FotMob'
[void](New-Item -Path $root -ItemType Directory -Force)
$globalPath=Join-Path (Split-Path $PSScriptRoot -Parent) 'Goal70_2026-11-02/Europe_FotMob/all_leagues_today.json'
$global=Get-Content -LiteralPath $globalPath -Raw -Encoding UTF8|ConvertFrom-Json
$defs=@(
 @('eng.1','ENG','Premier League'),@('ita.1','ITA','Serie A'),@('esp.1','ESP','LaLiga'),
 @('fra.1','FRA','Ligue 1'),@('por.1','POR','Liga Portugal'),@('tur.1','TUR','Super Lig'),
 @('gre.1','GRE','Super League'),@('pol.1','POL','Ekstraklasa'),@('den.1','DEN','Superligaen'),
 @('swe.1','SWE','Allsvenskan'),@('rou.1','ROU','Superliga'),@('srb.1','SRB','Super Liga'),
 @('esp.2','ESP','LaLiga2'),@('fra.2','FRA','Ligue 2'),@('tur.2','TUR','1. Lig')
)
function Fetch([string]$path){
 $url='https://www.fotmob.com/api/data/'+$path
 $r=@(& curl.exe --compressed -L --max-time 22 -sS -w '\nHTTP_STATUS:%{http_code}' $url 2>$null)
 $raw=$r -join "`n";$ix=$raw.LastIndexOf('HTTP_STATUS:')
 if($LASTEXITCODE -ne 0 -or $ix -lt 0){return @{ok=$false;error='transport_error';source=$url}}
 $code=$raw.Substring($ix+12).Trim()
 if($code -ne '200'){return @{ok=$false;error=('http_'+$code);source=$url}}
 try{$d=$raw.Substring(0,$ix).Trim()|ConvertFrom-Json}catch{return @{ok=$false;error='invalid_json';source=$url}}
 @{ok=$true;data=$d;source=$url}
}
function Fixture($m,[string]$slug){
 [pscustomobject]@{id=[string]$m.id;league=$slug;date=$m.status.utcTime;home_id=[string]$m.home.id;home=$m.home.name;
 away_id=[string]$m.away.id;away=$m.away.name;finished=$m.status.finished;started=$m.status.started;
 cancelled=$m.status.cancelled;awarded=$m.status.awarded;score=$m.status.scoreStr;
 source=('https://www.fotmob.com'+$m.pageUrl)}
}
function Extract-Corner($d,[string]$period){
 $c=@($d.content.stats.Periods.$period.stats|ForEach-Object {$_.stats}|Where-Object key -eq 'corners')
 if($c.Count -ne 1 -or @($c[0].stats).Count -ne 2){return $null}
 if($null -eq $c[0].stats[0] -or $null -eq $c[0].stats[1]){return $null}
 if([string]$c[0].stats[0] -notmatch '^\d+$' -or [string]$c[0].stats[1] -notmatch '^\d+$'){return $null}
 [pscustomobject]@{home=[int]$c[0].stats[0];away=[int]$c[0].stats[1];key=$c[0].key;period=$period}
}
for($i=0;$i -lt $defs.Count;$i++){
 if(($i%$ShardCount) -ne $Shard){continue}
 $def=$defs[$i];$slug=$def[0]
 $todayLeague=@($global|Where-Object {$_.country -eq $def[1] -and $_.name -eq $def[2]})
 if($todayLeague.Count -ne 1){throw ('Daily league identity mismatch '+$slug)}
 $first=$todayLeague[0].matches[0]
 $preview=Fetch ('matchDetails?matchId='+$first.id)
 if(-not $preview.ok){Write-Output "$slug preview failed";continue}
 $parent=$preview.data.general.parentLeagueId
 $season=if($slug -eq 'swe.1'){'2026'}else{'2026/2027'}
 $seasonQuery=if($slug -eq 'swe.1'){'2026'}else{'2026%2F2027'}
 $schedule=Fetch ('leagues?id='+$parent+'&ccode3=ROU&season='+$seasonQuery)
 if(-not $schedule.ok){Write-Output "$slug schedule failed";continue}
 if($schedule.data.details.selectedSeason -ne $season){throw ('Wrong season '+$slug)}
 $all=@($schedule.data.fixtures.allMatches|ForEach-Object {Fixture $_ $slug})
 $today=@($all|Where-Object {$_.date -and $_.date.Substring(0,10) -eq '2026-11-02'})
 if($today.Count -ne @($todayLeague[0].matches).Count){throw ('Daily fixture count mismatch '+$slug)}
 $ids=@($today|ForEach-Object {$_.home_id;$_.away_id}|Select-Object -Unique)
 $past=@($all|Where-Object {$_.finished -eq $true -and $_.cancelled -ne $true -and $_.awarded -ne $true -and
     $_.date.Substring(0,10) -lt '2026-11-02' -and ($_.home_id -in $ids -or $_.away_id -in $ids)})
 [pscustomobject]@{league=$slug;country=$def[1];name=$def[2];season=$season;parent_id=$parent;collected_at_utc=[DateTime]::UtcNow.ToString('o');
  source=$schedule.source;todays_fixtures=$today;expected_history=$past;min_season_matches=5;min_venue_matches=3}|
  ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $root ($slug+'.json')) -Encoding UTF8
 $count=0
 foreach($m in $past){
  $dest=Join-Path $root ('match_'+$slug+'_'+$m.id+'.json')
  if(Test-Path -LiteralPath $dest){continue}
  $r=Fetch ('matchDetails?matchId='+$m.id)
  $record=[ordered]@{fixture=$m;ok=$false;error=$null;source=$r.source;collected_at_utc=[DateTime]::UtcNow.ToString('o');
   period_definition='content.stats.Periods.FirstHalf, key corners; includes first-half added time';first_half=$null;second_half=$null;full_time=$null;reconciled=$false}
  if(-not $r.ok){$record.error=$r.error}else{
   $d=$r.data
   if([string]$d.general.matchId -ne $m.id -or [string]$d.general.homeTeam.id -ne $m.home_id -or [string]$d.general.awayTeam.id -ne $m.away_id){throw ('Match identity mismatch '+$m.id)}
   $record.first_half=Extract-Corner $d 'FirstHalf'
   $record.second_half=Extract-Corner $d 'SecondHalf'
   $record.full_time=Extract-Corner $d 'All'
   if($null -eq $record.first_half){$record.error='first_half_corners_missing'}
   elseif($null -eq $record.second_half -or $null -eq $record.full_time){$record.error='cannot_reconcile_periods'}
   elseif($record.first_half.home+$record.second_half.home -ne $record.full_time.home -or $record.first_half.away+$record.second_half.away -ne $record.full_time.away){$record.error='period_sum_mismatch'}
   else{$record.ok=$true;$record.reconciled=$true}
  }
  $record|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $dest -Encoding UTF8
  $count++;if(($count%20) -eq 0){Write-Output "$slug $count / $($past.Count)"}
 }
 Write-Output "$slug done, expected history $($past.Count), fixtures $($today.Count)"
}
