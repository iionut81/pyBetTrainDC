param([string]$DataDirectory = (Join-Path $PSScriptRoot 'Europe_ESPN'))
$ErrorActionPreference = 'Stop'
$history = New-Object System.Collections.Generic.List[object]
$rejected = New-Object System.Collections.Generic.List[object]
$fixtures = New-Object System.Collections.Generic.List[object]
$coverage = New-Object System.Collections.Generic.List[object]
$seen = @{}
foreach ($file in (Get-ChildItem -LiteralPath $DataDirectory -Filter '*.json' | Where-Object {$_.Name -match '^[a-z]{3}\.[12]\.json$'})) {
    $data = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $coverage.Add([pscustomobject]@{league=$data.league;country=$data.country;uefa_rank=$data.uefa_2026_rank;
        today_ok=$data.today_ok;today_error=$data.today_error;todays_fixtures=@($data.todays_events).Count;
        history_ok=$data.history_ok;historical_events=@($data.historical_events).Count;history_error=$data.history_error})
    foreach ($fixture in $data.todays_events) { $fixtures.Add($fixture) }
    foreach ($event in $data.historical_events) {
        $key="$($event.league)|$($event.id)"
        if ($seen.ContainsKey($key)) { throw "Duplicate historical ID: $key" }; $seen[$key]=$true
        $h=@($event.teams | Where-Object {$_.homeAway -eq 'home'})
        $a=@($event.teams | Where-Object {$_.homeAway -eq 'away'})
        $reason=$null
        if ($h.Count -ne 1 -or $a.Count -ne 1) { $reason='invalid_competitors' }
        elseif ([string]$h[0].score -notmatch '^\d+$' -or [string]$a[0].score -notmatch '^\d+$') { $reason='missing_scores' }
        $parsed=New-Object System.Collections.Generic.List[object]
        if (-not $reason) {
            foreach ($g in $event.goals) {
                if ($g.minute -notmatch '^(\d{1,3})\D*(?:\+(\d+)\D*)?$') { $reason='invalid_goal_minute'; break }
                $minute=[int]$Matches[1]
                if ($minute -gt 90 -or ($g.team_id -ne $h[0].id -and $g.team_id -ne $a[0].id)) { $reason='extra_time_or_unknown_team';break }
                $parsed.Add([pscustomobject]@{minute=$minute;display=$g.minute;team=$g.team_id;own_goal=$g.own_goal})
            }
        }
        $mapping='provider_credited_team'
        if (-not $reason) {
            $hGoals=@($parsed | Where-Object {$_.team -eq $h[0].id}).Count
            $aGoals=@($parsed | Where-Object {$_.team -eq $a[0].id}).Count
            if ($hGoals -ne [int]$h[0].score -or $aGoals -ne [int]$a[0].score) {
                $own=@($parsed | Where-Object {$_.own_goal})
                if ($own.Count -gt 0) {
                    foreach ($g in $own) { $g.team=if ($g.team -eq $h[0].id) {$a[0].id} else {$h[0].id} }
                    $hGoals=@($parsed | Where-Object {$_.team -eq $h[0].id}).Count
                    $aGoals=@($parsed | Where-Object {$_.team -eq $a[0].id}).Count
                    $mapping='own_goal_credit_flipped_to_reconcile'
                }
                if ($hGoals -ne [int]$h[0].score -or $aGoals -ne [int]$a[0].score) { $reason='goal_list_score_mismatch' }
            }
        }
        if ($reason) { $rejected.Add([pscustomobject]@{league=$event.league;id=$event.id;date=$event.date;name=$event.name;reason=$reason;source=$event.source});continue }
        $h70=@($parsed | Where-Object {$_.team -eq $h[0].id -and $_.minute -le 70}).Count
        $a70=@($parsed | Where-Object {$_.team -eq $a[0].id -and $_.minute -le 70}).Count
        $first=@($parsed | Sort-Object minute | Select-Object -First 1)
        $history.Add([pscustomobject]@{league=$event.league;id=$event.id;date=$event.date;season=2026;
            home_id=$h[0].id;home=$h[0].name;away_id=$a[0].id;away=$a[0].name;
            home_score=[int]$h[0].score;away_score=[int]$a[0].score;home70=$h70;away70=$a70;
            first_goal=if ($first.Count) {$first[0].display} else {''};success70=($h70+$a70 -gt 0);
            home_goals=(@($parsed | Where-Object {$_.team -eq $h[0].id} | ForEach-Object {$_.display}) -join ';');
            away_goals=(@($parsed | Where-Object {$_.team -eq $a[0].id} | ForEach-Object {$_.display}) -join ';');
            goal_mapping=$mapping;source=$event.source})
    }
}
function Get-Stats([string]$TeamId,[string]$League,[string]$Venue='all') {
    $sample=@($history | Where-Object {$_.league -eq $League -and
        (($Venue -eq 'home' -and $_.home_id -eq $TeamId) -or
         ($Venue -eq 'away' -and $_.away_id -eq $TeamId) -or
         ($Venue -eq 'all' -and ($_.home_id -eq $TeamId -or $_.away_id -eq $TeamId)))})
    $gf=0;$ga=0;$wins=0
    foreach ($m in $sample) {if($m.home_id -eq $TeamId){$gf+=$m.home70;$ga+=$m.away70}else{$gf+=$m.away70;$ga+=$m.home70};if($m.success70){$wins++}}
    [pscustomobject]@{n=$sample.Count;gf=$gf;ga=$ga;wins=$wins}
}
function Get-Lambda($A,$B) { ($A.gf/$A.n+$B.ga/$B.n+$B.gf/$B.n+$A.ga/$A.n)/2 }
$models=New-Object System.Collections.Generic.List[object]
foreach ($f in $fixtures) {
    $h=@($f.teams | Where-Object {$_.homeAway -eq 'home'})[0]
    $a=@($f.teams | Where-Object {$_.homeAway -eq 'away'})[0]
    if ($f.state -ne 'pre' -or $f.time_valid -ne $true) {continue}
    $hs=Get-Stats $h.id $f.league; $as=Get-Stats $a.id $f.league
    $hv=Get-Stats $h.id $f.league 'home';$av=Get-Stats $a.id $f.league 'away'
    $bad=@($rejected | Where-Object {$_.league -eq $f.league -and ($_.name -match [regex]::Escape($h.name) -or $_.name -match [regex]::Escape($a.name))})
    $eligible=($hs.n -ge 5 -and $as.n -ge 5 -and $hv.n -ge 3 -and $av.n -ge 3 -and $bad.Count -eq 0)
    $ls=$null;$lv=$null;$lc=$null;$p=$null;$stress=$null;$fair=$null
    if ($eligible) { $ls=Get-Lambda $hs $as; $lv=Get-Lambda $hv $av;$lc=($ls+$lv)/2;$p=1-[math]::Exp(-$lc);$stress=1-[math]::Exp(-0.8*$lc);$fair=1/$p }
    $models.Add([pscustomobject]@{league=$f.league;id=$f.id;date=$f.date;home=$h.name;away=$a.name;
        home_id=$h.id;away_id=$a.id;eligible=$eligible;home_n=$hs.n;home_gf70=$hs.gf;home_ga70=$hs.ga;home_success=$hs.wins;
        away_n=$as.n;away_gf70=$as.gf;away_ga70=$as.ga;away_success=$as.wins;
        home_venue_n=$hv.n;home_venue_gf70=$hv.gf;home_venue_ga70=$hv.ga;home_venue_success=$hv.wins;
        away_venue_n=$av.n;away_venue_gf70=$av.gf;away_venue_ga70=$av.ga;away_venue_success=$av.wins;
        rejected_team_matches=$bad.Count;lambda_season=$ls;lambda_venue=$lv;lambda_combined=$lc;p_raw=$p;p_stress=$stress;fair_odds=$fair;source=$f.source})
}
$history | Export-Csv -LiteralPath (Join-Path $DataDirectory 'historical_validated.csv') -NoTypeInformation -Encoding UTF8
$rejected | Export-Csv -LiteralPath (Join-Path $DataDirectory 'historical_rejected.csv') -NoTypeInformation -Encoding UTF8
$coverage | Export-Csv -LiteralPath (Join-Path $DataDirectory 'league_coverage.csv') -NoTypeInformation -Encoding UTF8
$fixtures | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $DataDirectory 'today_fixtures.json') -Encoding UTF8
$models | Sort-Object p_raw -Descending | Export-Csv -LiteralPath (Join-Path $DataDirectory 'all_models.csv') -NoTypeInformation -Encoding UTF8
$models | Sort-Object p_raw -Descending | Select-Object league,home,away,eligible,home_n,away_n,rejected_team_matches,p_raw,p_stress,fair_odds | ConvertTo-Json -Depth 4
"Validated historical matches: $($history.Count); rejected: $($rejected.Count); fixtures evaluated: $($models.Count)"
