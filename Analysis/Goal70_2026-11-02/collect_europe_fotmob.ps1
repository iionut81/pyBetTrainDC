param([int]$Shard=0,[int]$ShardCount=1,[string]$AsOf='2026-11-02')
$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot 'Europe_FotMob'
[void](New-Item -Path $root -ItemType Directory -Force)
function Read-FotMob([string]$Path) {
    $url='https://www.fotmob.com/api/data/'+$Path
    $r=@(& curl.exe --compressed -L --max-time 25 -sS -w '\nHTTP_STATUS:%{http_code}' $url 2>$null)
    if($LASTEXITCODE -ne 0){return @{ok=$false;error='curl_failed';url=$url}}
    $t=$r -join "`n";$ix=$t.LastIndexOf('HTTP_STATUS:')
    if($ix -lt 0){return @{ok=$false;error='missing_http_status';url=$url}}
    $status=$t.Substring($ix+12).Trim()
    if($status -ne '200'){return @{ok=$false;error="http_$status";url=$url}}
    try{$d=$t.Substring(0,$ix).Trim()|ConvertFrom-Json}catch{return @{ok=$false;error='invalid_json';url=$url}}
    if($null -eq $d){return @{ok=$false;error='empty_json';url=$url}}
    @{ok=$true;data=$d;url=$url}
}
function Select-Fixture($m,[string]$league){
    [pscustomobject]@{league=$league;id=[string]$m.id;date=$m.status.utcTime;
        home_id=[string]$m.home.id;home=$m.home.name;away_id=[string]$m.away.id;away=$m.away.name;
        finished=$m.status.finished;started=$m.status.started;cancelled=$m.status.cancelled;
        awarded=$m.status.awarded;score=$m.status.scoreStr;round=$m.round;
        source=('https://www.fotmob.com'+$m.pageUrl)}
}
$defs=@(@('pol.1',196,'Poland',12),@('rou.1',189,'Romania',0),@('srb.1',182,'Serbia',0),@('tur.2',165,'Turkey second tier',0))
if($Shard -eq 0){
    $today=Read-FotMob ('matches?date='+$AsOf.Replace('-',''))
    if(-not $today.ok){throw 'Unable to read global fixture list'}
    $projected=@($today.data.leagues|ForEach-Object{[pscustomobject]@{id=$_.id;name=$_.name;country=$_.ccode;
        matches=@($_.matches|ForEach-Object{[pscustomobject]@{id=$_.id;home=$_.home;away=$_.away;status=$_.status;pageUrl=$_.pageUrl}})}})
    $projected|ConvertTo-Json -Depth 9|Set-Content -LiteralPath (Join-Path $root 'all_leagues_today.json') -Encoding UTF8
}
for($i=0;$i -lt $defs.Count;$i++){
    if(($i%$ShardCount) -ne $Shard){continue}
    $def=$defs[$i];$slug=[string]$def[0]
    $leagueResult=Read-FotMob ("leagues?id=$($def[1])&ccode3=ROU&season=2026%2F2027")
    if(-not $leagueResult.ok){Write-Output "$slug league error: $($leagueResult.error)";continue}
    $all=@($leagueResult.data.fixtures.allMatches)
    if($leagueResult.data.details.selectedSeason -ne '2026/2027' -or $all.Count -lt 1){throw "Wrong season or empty fixture list: $slug"}
    $today=@($all|Where-Object{$_.status.utcTime -and $_.status.utcTime.Substring(0,10) -eq $AsOf}|ForEach-Object{Select-Fixture $_ $slug})
    $ids=@($today|ForEach-Object{$_.home_id;$_.away_id}|Select-Object -Unique)
    $prior=@($all|Where-Object{$_.status.finished -eq $true -and $_.status.utcTime.Substring(0,10) -lt $AsOf -and
        ([string]$_.home.id -in $ids -or [string]$_.away.id -in $ids)}|ForEach-Object{Select-Fixture $_ $slug})
    [pscustomobject]@{league=$slug;country=$def[2];uefa_2026_rank=$def[3];as_of=$AsOf;
        league_source=$leagueResult.url;league_name=$leagueResult.data.details.name;season=$leagueResult.data.details.selectedSeason;
        todays_fixtures=$today;all_season_fixtures=@($all|ForEach-Object{Select-Fixture $_ $slug})} |
        ConvertTo-Json -Depth 9|Set-Content -LiteralPath (Join-Path $root "$slug.json") -Encoding UTF8
    $done=0
    foreach($m in $prior){
        $file=Join-Path $root ("match_"+$slug+'_'+$m.id+'.json')
        if(Test-Path -LiteralPath $file){continue}
        $r=Read-FotMob ('matchDetails?matchId='+$m.id)
        if(-not $r.ok){[pscustomobject]@{fixture=$m;ok=$false;error=$r.error}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $file -Encoding UTF8;continue}
        $d=$r.data
        $goals=New-Object System.Collections.Generic.List[object]
        foreach($side in @('home','away')){
            $map=if($side -eq 'home'){$d.header.events.homeTeamGoals}else{$d.header.events.awayTeamGoals}
            foreach($prop in $map.PSObject.Properties){foreach($g in @($prop.Value)){
                $goals.Add([pscustomobject]@{side=$side;time=$g.time;added_time=$g.overloadTime;time_str=$g.timeStr;
                    type=$g.type;event_id=$g.eventId;home_score=$g.homeScore;away_score=$g.awayScore;
                    goal_description=$g.goalDescription;period=$g.shotmapEvent.period;is_own_goal=$g.shotmapEvent.isOwnGoal})
            }}
        }
        [pscustomobject]@{fixture=$m;ok=$true;source=$r.url;general=($d.general|Select-Object matchId,parentLeagueId,leagueName,homeTeam,awayTeam,started,finished);
            header_teams=@($d.header.teams|Select-Object id,name,score);header_status=$d.header.status;goals=$goals;
            event_types=@($d.content.matchFacts.events.events|Group-Object type|Select-Object Name,Count)} |
            ConvertTo-Json -Depth 10|Set-Content -LiteralPath $file -Encoding UTF8
        $done++;if(($done%10) -eq 0){Write-Output "$slug collected $done / $($prior.Count)"}
    }
    Write-Output "$slug complete: fixtures=$($today.Count), target historical=$($prior.Count), newly fetched=$done"
}
