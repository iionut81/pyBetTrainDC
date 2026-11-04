#requires -Version 5.1
[CmdletBinding()]
param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$baseUrl = 'http://127.0.0.1:18770'
$pageUrl = $baseUrl + '/Previews/Pred_Research_Terminal/dist/index.html'
function Get-PredHealth {
    try { return Invoke-RestMethod -Uri ($baseUrl + '/api/health') -TimeoutSec 1 }
    catch { return $null }
}
$health = Get-PredHealth
if ($health -and ($health.app -ne 'pred-dashboard' -or $health.root -ne $projectDir)) { throw 'Port 18770 belongs to another project. Refusing to reuse it.' }
if (-not $health) {
    $nodeExe = (Get-Command node -ErrorAction Stop).Source
    $serverScript = Join-Path $PSScriptRoot 'server.mjs'
    $runtimeDir = Join-Path $PSScriptRoot '.runtime'
    [IO.Directory]::CreateDirectory($runtimeDir) | Out-Null
    $serverProcess = Start-Process -FilePath $nodeExe -ArgumentList ('"' + $serverScript + '"') -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'server.out.log') -RedirectStandardError (Join-Path $runtimeDir 'server.err.log')
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 150
        $health = Get-PredHealth
        if ($health) { break }
        if ($serverProcess.HasExited) { throw 'Dashboard service could not start. See .runtime/server.err.log.' }
    }
}
if (-not $health -or $health.app -ne 'pred-dashboard' -or $health.root -ne $projectDir) { throw 'Dashboard service is unavailable.' }
if (-not $NoBrowser) {
    $chromeExe = @((Join-Path $env:LOCALAPPDATA 'Google/Chrome/Application/chrome.exe'),'C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Google/Chrome/Application/chrome.exe') | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $chromeExe) { throw 'Google Chrome is not installed.' }
    Start-Process -FilePath $chromeExe -ArgumentList @('--new-window',('"' + $pageUrl + '"'))
}
[pscustomobject]@{status='READY';url=$pageUrl;browser='Chrome';project=$projectDir} | ConvertTo-Json
