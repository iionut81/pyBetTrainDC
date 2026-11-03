param([int]$Shard=0,[int]$ShardCount=4)
$ErrorActionPreference='Stop'
$out=Join-Path $PSScriptRoot 'ESPN';[void](New-Item $out -ItemType Directory -Force)
$root=Split-Path $PSScriptRoot -Parent
$es=@(Import-Csv (Join-Path $root 'Goal70_2026-11-03/Data/historical_validated.csv') -Encoding UTF8|Where-Object {$_.league -eq 'eng.2' -and ($_.home_id -in @('374','381') -or $_.away_id -in @('374','381'))}|Sort-Object id)
$fm=@(Import-Csv (Join-Path $PSScriptRoot 'historical_corners_1h.csv') -Encoding UTF8)
for($i=0;$i -lt $es.Count;$i++){
 if($i%$ShardCount -ne $Shard){continue};$m=$es[$i];$dest=Join-Path $out ('match_'+$m.id+'.json')
 if(Test-Path -LiteralPath $dest){continue}
 $url='https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2/summary?event='+$m.id
 $d=(& curl.exe --compressed -L --max-time 25 --retry 1 -sS $url)|ConvertFrom-Json
 if($LASTEXITCODE -ne 0 -or [string]$d.header.id -ne $m.id){throw ('ESPN identity failed '+$m.id)}
 $plays=@($d.commentary|Where-Object {$_.play.id}|ForEach-Object {$_.play}|Sort-Object id -Unique)
 $corners=@($plays|Where-Object {$_.type.type -eq 'corner-awarded'}|ForEach-Object {[pscustomobject]@{id=$_.id;period=[int]$_.period.number;minute=$_.clock.displayValue;seconds=$_.clock.value;team=$_.team.displayName}})
 $red=@($plays|Where-Object {$_.type.type -match 'red-card|second-yellow|yellow-red'}|ForEach-Object {[pscustomobject]@{id=$_.id;type=$_.type.type;period=[int]$_.period.number;minute=$_.clock.displayValue;seconds=$_.clock.value;team=$_.team.displayName}})
 $match=@($fm|Where-Object {$_.league -eq 'eng.2' -and $_.date.Substring(0,10) -eq $m.date.Substring(0,10) -and $_.home -eq $m.home -and $_.away -eq $m.away})
 if($match.Count -ne 1){throw ('Provider crosswalk conflict '+$m.id)};$f=$match[0]
 $ft=New-Object System.Collections.Generic.List[object]
 foreach($side in @('home','away')){
  $team=[string]$m.$side;$stats=@($d.boxscore.teams|Where-Object {$_.team.displayName -eq $team})
  $stat=@($stats.statistics|Where-Object name -eq 'wonCorners')
  if($stat.Count -ne 1){throw ('FT corner stat missing '+$m.id+' '+$side)}
  $n1=@($corners|Where-Object {$_.period -eq 1 -and $_.team -eq $team}).Count
  $n2=@($corners|Where-Object {$_.period -eq 2 -and $_.team -eq $team}).Count
  $reported=[int]$stat[0].displayValue
  $ft.Add([pscustomobject]@{side=$side;team=$team;first_half=$n1;second_half=$n2;full_time=$reported;timeline_complete=($n1+$n2 -eq $reported);fotmob_agrees=($n1 -eq [int]$f.($side+'_corners_1h') -and $n2 -eq [int]$f.($side+'_corners_2h') -and $reported -eq [int]$f.($side+'_corners_ft'))})
 }
 $record=[pscustomobject]@{id=$m.id;fotmob_id=$f.id;date=$m.date;home=$m.home;away=$m.away;source=$url;commentary_count=@($d.commentary).Count;first_half_end_present=(@($d.commentary|Where-Object text -match 'First Half ends').Count -gt 0);corners=$corners;red_cards=$red;period_counts=$ft;all_periods_agree=(@($ft|Where-Object {$_.timeline_complete -ne $true -or $_.fotmob_agrees -ne $true}).Count -eq 0);fetched_at_utc=[DateTime]::UtcNow.ToString('o')}
 $record|ConvertTo-Json -Depth 7|Set-Content $dest -Encoding UTF8
}
Write-Output ('ESPN shard '+$Shard+' done')
