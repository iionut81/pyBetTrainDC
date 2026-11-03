param([int]$Shard = 0, [int]$ShardCount = 1, [string]$AsOf = '2026-11-02')
$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot 'Europe_ESPN'
[void](New-Item -Path $root -ItemType Directory -Force)
$map = @(
    @('eng.1','England',1), @('ita.1','Italy',2), @('esp.1','Spain',3),
    @('ger.1','Germany',4), @('fra.1','France',5), @('por.1','Portugal',6),
    @('ned.1','Netherlands',7), @('bel.1','Belgium',8), @('tur.1','Turkey',9),
    @('cze.1','Czechia',10), @('gre.1','Greece',11), @('pol.1','Poland',12),
    @('den.1','Denmark',13), @('nor.1','Norway',14), @('cyp.1','Cyprus',15),
    @('sui.1','Switzerland',16), @('aut.1','Austria',17), @('sco.1','Scotland',18),
    @('swe.1','Sweden',19), @('cro.1','Croatia',20),
    @('rou.1','Romania',0), @('srb.1','Serbia',0),
    @('eng.2','England second tier',0), @('ger.2','Germany second tier',0),
    @('esp.2','Spain second tier',0), @('ita.2','Italy second tier',0),
    @('fra.2','France second tier',0), @('bel.2','Belgium second tier',0),
    @('tur.2','Turkey second tier',0)
)
function Read-PublicJson([string]$Url) {
    if ($Url -notmatch '^https://site\.api\.espn\.com/apis/site/v2/sports/soccer/') { throw 'Unexpected host' }
    $response = @(& curl.exe --compressed -L --max-time 30 -sS -w '\nHTTP_STATUS:%{http_code}' $Url 2>$null)
    if ($LASTEXITCODE -ne 0) { return @{ok=$false; error='curl_failed';url=$Url} }
    $raw = $response -join "`n"
    $index = $raw.LastIndexOf('HTTP_STATUS:')
    if ($index -lt 0) { return @{ok=$false;error='no_http_status';url=$Url} }
    $status = $raw.Substring($index + 12).Trim()
    if ($status -ne '200') { return @{ok=$false;error="http_$status";url=$Url} }
    try { $parsed = $raw.Substring(0,$index).Trim() | ConvertFrom-Json }
    catch { return @{ok=$false;error='invalid_json';url=$Url} }
    return @{ok=$true;data=$parsed;url=$Url}
}
function Select-PublicEvent($Event, [string]$League) {
    $c = $Event.competitions | Select-Object -First 1
    $teams = @($c.competitors | ForEach-Object {
        [pscustomobject]@{ id=[string]$_.id; name=$_.team.displayName; homeAway=$_.homeAway; score=$_.score }
    })
    $goals = @($c.details | Where-Object {$_.scoringPlay -eq $true -and $_.shootout -ne $true} | ForEach-Object {
        [pscustomobject]@{ minute=$_.clock.displayValue; clock_seconds=$_.clock.value;
            team_id=[string]$_.team.id; own_goal=[bool]$_.ownGoal; type_id=$_.type.id; score_value=$_.scoreValue }
    })
    [pscustomobject]@{ id=[string]$Event.id; league=$League; date=$Event.date;
        season_year=$Event.season.year; season_slug=$Event.season.slug; name=$Event.name;
        time_valid=$c.timeValid; state=$c.status.type.state; status=$c.status.type.name;
        completed=$c.status.type.completed; venue=$c.venue.fullName; teams=$teams;
        goals=$goals; details_present=($null -ne $c.details);
        source="https://www.espn.com/soccer/match/_/gameId/$($Event.id)" }
}
$index = -1
foreach ($entry in $map) {
    $index++
    if (($index % $ShardCount) -ne $Shard) { continue }
    $slug = [string]$entry[0]
    $record = [ordered]@{league=$slug;country=$entry[1];uefa_2026_rank=$entry[2];as_of=$AsOf;
        collected_at_utc=[DateTime]::UtcNow.ToString('o'); today_ok=$false; history_ok=$false;
        today_error=$null;history_error=$null;todays_events=@();historical_events=@()}
    $todayUrl = "https://site.api.espn.com/apis/site/v2/sports/soccer/$slug/scoreboard?dates=$($AsOf.Replace('-',''))&limit=1000"
    $today = Read-PublicJson $todayUrl
    if ($today.ok -and $today.data.leagues[0].slug -eq $slug) {
        $record.today_ok=$true
        $record.todays_events=@($today.data.events | ForEach-Object {Select-PublicEvent $_ $slug})
    } else { $record.today_error = if ($today.ok) {'wrong_or_missing_league'} else {$today.error} }
    $record.today_source=$todayUrl
    if (@($record.todays_events | Where-Object {$_.state -eq 'pre'}).Count -gt 0) {
        $historyUrl = "https://site.api.espn.com/apis/site/v2/sports/soccer/$slug/scoreboard?dates=2026&limit=1000"
        $history = Read-PublicJson $historyUrl
        if ($history.ok -and $history.data.leagues[0].slug -eq $slug) {
            $record.history_ok=$true
            $record.historical_events=@($history.data.events | Where-Object {
                $_.season.year -eq 2026 -and $_.date.Substring(0,10) -lt $AsOf -and
                $_.competitions[0].status.type.completed -eq $true
            } | ForEach-Object {Select-PublicEvent $_ $slug})
            $record.returned_year_events=@($history.data.events).Count
        } else { $record.history_error = if ($history.ok) {'wrong_or_missing_league'} else {$history.error} }
        $record.history_source=$historyUrl
    }
    $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $root "$slug.json") -Encoding UTF8
    Write-Output ("{0}: today={1} fixtures={2} history={3} error={4}/{5}" -f $slug,$record.today_ok,@($record.todays_events).Count,@($record.historical_events).Count,$record.today_error,$record.history_error)
}
