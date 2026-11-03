$ErrorActionPreference='Stop'
$espn=Join-Path $PSScriptRoot 'Europe_ESPN'
$fm=Join-Path $PSScriptRoot 'Europe_FotMob'
$dest=Join-Path $PSScriptRoot 'Europe_Combined'
[void](New-Item -Path $dest -ItemType Directory -Force)
Get-ChildItem -LiteralPath $espn -Filter '*.json'|Where-Object{$_.Name -match '^[a-z]{3}\.[12]\.json$'}|ForEach-Object{
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dest $_.Name) -Force
}
foreach($slug in @('pol.1','rou.1','srb.1','tur.2')){
    $definition=Get-Content -LiteralPath (Join-Path $fm "$slug.json") -Raw -Encoding UTF8|ConvertFrom-Json
    $past=New-Object System.Collections.Generic.List[object]
    foreach($file in (Get-ChildItem -LiteralPath $fm -Filter ("match_"+$slug+'_*.json'))){
        $d=Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8|ConvertFrom-Json
        $f=$d.fixture
        if(-not $d.ok){
            $past.Add([pscustomobject]@{league=$slug;id=$f.id;date=$f.date;name=($f.away+' at '+$f.home);
                teams=@();goals=@();source=$f.source;fetch_error=$d.error});continue
        }
        if([string]$d.general.matchId -ne [string]$f.id -or
           [string]$d.general.homeTeam.id -ne [string]$f.home_id -or
           [string]$d.general.awayTeam.id -ne [string]$f.away_id){throw "Fixture identity mismatch $($f.id)"}
        if($f.score -notmatch '^(\d+)\s*-\s*(\d+)$'){throw "Unknown full-time score $($f.id)"}
        $hScore=[int]$Matches[1];$aScore=[int]$Matches[2]
        $teams=@([pscustomobject]@{id=$f.home_id;name=$f.home;homeAway='home';score=[string]$hScore},
                 [pscustomobject]@{id=$f.away_id;name=$f.away;homeAway='away';score=[string]$aScore})
        $goals=@($d.goals|Where-Object{$_.type -eq 'Goal'}|ForEach-Object{
            $display=[string]$_.time+"'"
            if($_.added_time -gt 0){$display+='+'+[string]$_.added_time+"'"}
            [pscustomobject]@{minute=$display;team_id=if($_.side -eq 'home'){$f.home_id}else{$f.away_id};
                own_goal=$_.is_own_goal;type_id='Goal';score_value=1;period=$_.period}
        })
        $past.Add([pscustomobject]@{league=$slug;id=$f.id;date=$f.date;season_year=2026;season_slug='2026-2027';
            name=($f.away+' at '+$f.home);time_valid=$true;state='post';status='STATUS_FULL_TIME';
            completed=$true;teams=$teams;goals=$goals;details_present=$true;source=$f.source})
    }
    $future=@($definition.todays_fixtures|ForEach-Object{
        [pscustomobject]@{league=$slug;id=$_.id;date=$_.date;season_year=2026;season_slug='2026-2027';
            name=($_.away+' at '+$_.home);time_valid=$true;
            state=if($_.finished){'post'}elseif($_.started){'in'}else{'pre'};
            status=if($_.cancelled){'STATUS_CANCELLED'}else{'STATUS_SCHEDULED'};completed=$_.finished;
            teams=@([pscustomobject]@{id=$_.home_id;name=$_.home;homeAway='home';score='0'},
                    [pscustomobject]@{id=$_.away_id;name=$_.away;homeAway='away';score='0'});
            goals=@();source=$_.source}
    })
    [pscustomobject]@{league=$slug;country=$definition.country;uefa_2026_rank=$definition.uefa_2026_rank;as_of='2026-11-02';
        today_ok=$true;history_ok=$true;today_error=$null;history_error=$null;
        todays_events=$future;historical_events=$past;today_source=$definition.league_source}|
        ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $dest "$slug.json") -Encoding UTF8
    Write-Output "$slug merged: historical=$($past.Count), fixtures=$($future.Count)"
}
