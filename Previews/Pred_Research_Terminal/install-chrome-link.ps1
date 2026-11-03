#requires -Version 5.1
[CmdletBinding()]
param([switch]$PreviewOnly)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$indexFile = Join-Path $PSScriptRoot 'dist/index.html'
if (-not (Test-Path -LiteralPath $indexFile)) { throw 'Dashboard index.html not found.' }
$chromeCandidates = @(
    (Join-Path $env:LOCALAPPDATA 'Google/Chrome/Application/chrome.exe'),
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
)
$chromeExe = [string]($chromeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
if (-not $chromeExe) { throw 'Google Chrome is not installed.' }
$fileUri = ([uri]$indexFile).AbsoluteUri
$arguments = '--new-window "' + $indexFile + '"'
$command = '"' + $chromeExe + '" ' + $arguments
$relativeKey = 'Software\Classes\pred-dashboard'
$existing = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($relativeKey)
if ($existing) {
    try { if ($existing.GetValue('PredProject') -ne $projectDir) { throw 'Protocol already belongs to another application; refusing to replace it.' } }
    finally { $existing.Dispose() }
}
$shortcutPath = Join-Path $PSScriptRoot 'Deschide PRED in Chrome.url'
$result = [ordered]@{protocol='pred-dashboard://open';chrome_exe=$chromeExe;index_file=$indexFile;index_uri=$fileUri;command=$command;shortcut=$shortcutPath;changes_default_browser=$false;accepts_uri_arguments=$false}
if ($PreviewOnly) { $result | ConvertTo-Json; return }
# A fixed Chrome executable and fixed dashboard URI. No %1 or user URL data
# is interpolated into the command; other protocols/file associations are untouched.
$key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($relativeKey)
try {
    $key.SetValue('','URL:PRED Dashboard in Chrome',[Microsoft.Win32.RegistryValueKind]::String)
    $key.SetValue('URL Protocol','',[Microsoft.Win32.RegistryValueKind]::String)
    $key.SetValue('PredProject',$projectDir,[Microsoft.Win32.RegistryValueKind]::String)
    $icon = $key.CreateSubKey('DefaultIcon')
    try { $icon.SetValue('',('"'+$chromeExe+'",0')) } finally { $icon.Dispose() }
    $open = $key.CreateSubKey('shell\open\command')
    try { $open.SetValue('',$command,[Microsoft.Win32.RegistryValueKind]::String) } finally { $open.Dispose() }
} finally { $key.Dispose() }
$shortcutText = "[InternetShortcut]`r`nURL=pred-dashboard://open`r`nIconFile=$chromeExe`r`nIconIndex=0`r`n"
[IO.File]::WriteAllText($shortcutPath,$shortcutText,[Text.Encoding]::Unicode)
$verify = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($relativeKey+'\shell\open\command')
try { if ($verify.GetValue('') -ne $command) { throw 'Protocol command verification failed.' } }
finally { $verify.Dispose() }
if (-not ([IO.File]::ReadAllText($shortcutPath,[Text.Encoding]::Unicode)).Contains('URL=pred-dashboard://open')) { throw 'Chrome shortcut verification failed.' }
$result['installed'] = $true
$result['verified_at_utc'] = [DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'chrome-launcher.json'),($result | ConvertTo-Json),(New-Object Text.UTF8Encoding($false)))
$result | ConvertTo-Json
