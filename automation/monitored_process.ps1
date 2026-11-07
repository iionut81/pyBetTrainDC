# Stream child output to private files while it runs, even without line breaks.
# No scheduling or research is performed by this helper.
if (-not ('PredProcessStreams' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
public static class PredProcessStreams {
    [DllImport("kernel32.dll")] private static extern uint SetThreadExecutionState(uint flags);
    public static void KeepAwake(bool enabled) {
        if (SetThreadExecutionState(enabled ? 0x80000001u : 0x80000000u) == 0)
            throw new System.ComponentModel.Win32Exception();
    }
    public static async Task Pump(StreamReader source, string destination) {
        using (var file = new FileStream(destination, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
        using (var writer = new StreamWriter(file, new UTF8Encoding(false))) {
            var buffer = new char[2048];
            int count;
            while ((count = await source.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false)) > 0) {
                await writer.WriteAsync(buffer, 0, count).ConfigureAwait(false);
                await writer.FlushAsync().ConfigureAwait(false);
            }
        }
    }
}
'@
}

function Write-PredAtomicJson {
    param([string]$Path, $Value)
    $tempPath = $Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    try {
        [IO.File]::WriteAllText($tempPath, ($Value | ConvertTo-Json -Depth 10), (New-Object Text.UTF8Encoding($false)))
        for ($attempt = 0; ; $attempt++) {
            try {
                if ([IO.File]::Exists($Path)) { [IO.File]::Replace($tempPath, $Path, [System.Management.Automation.Language.NullString]::Value) }
                else { [IO.File]::Move($tempPath, $Path) }
                break
            } catch {
                $cause = $_.Exception.GetBaseException()
                if ($attempt -ge 5 -or ($cause -isnot [IO.IOException] -and $cause -isnot [UnauthorizedAccessException])) { throw }
                Start-Sleep -Milliseconds (50 * [math]::Pow(2, $attempt))
            }
        }
    } finally { if ([IO.File]::Exists($tempPath)) { [IO.File]::Delete($tempPath) } }
}

function Test-PredProcessIdentity {
    param($ProcessId, [string]$StartedAtUtc)
    if (-not $ProcessId -or -not $StartedAtUtc) { return $false }
    $existingProcess = Get-Process -Id ([int]$ProcessId) -ErrorAction SilentlyContinue
    if (-not $existingProcess) { return $false }
    try {
        return [Math]::Abs(($existingProcess.StartTime.ToUniversalTime() - [datetime]::Parse($StartedAtUtc).ToUniversalTime()).TotalSeconds) -lt 1
    } catch { return $false }
}

function Invoke-PredMonitoredProcess {
    param(
        [Diagnostics.ProcessStartInfo]$StartInfo,
        [string]$InputText,
        [string]$StdoutPath,
        [string]$StderrPath,
        [scriptblock]$OnHeartbeat,
        [int]$HeartbeatSeconds = 10,
        [int]$TimeoutSeconds = 13800,
        [switch]$AllowIdleSleep
    )
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $StartInfo
    $pumpOut = $null; $pumpError = $null; $awake = $false
    try {
        if (-not $AllowIdleSleep) { [PredProcessStreams]::KeepAwake($true); $awake = $true }
        if (-not $process.Start()) { throw 'Child process could not be started.' }
        $pumpOut = [PredProcessStreams]::Pump($process.StandardOutput, $StdoutPath)
        $pumpError = [PredProcessStreams]::Pump($process.StandardError, $StderrPath)
        $startedAt = [DateTime]::UtcNow
        $workerStartedAt = $process.StartTime.ToUniversalTime().ToString('o')
        $lastHeartbeat = [datetime]::MinValue
        # Both output streams are already drained while the child reads stdin.
        $inputBytes = (New-Object Text.UTF8Encoding($false)).GetBytes($InputText + "`n")
        $process.StandardInput.BaseStream.Write($inputBytes, 0, $inputBytes.Length)
        $process.StandardInput.Close()
        do {
            $now = [DateTime]::UtcNow
            if (($now - $lastHeartbeat).TotalSeconds -ge $HeartbeatSeconds) {
                $beat = [ordered]@{runner_pid=$PID;runner_started_at_utc=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o');worker_pid=$process.Id;worker_started_at_utc=$workerStartedAt;heartbeat_at_utc=$now.ToString('o');stdout_bytes=0;stderr_bytes=0}
                if (Test-Path -LiteralPath $StdoutPath) { $beat.stdout_bytes = (Get-Item -LiteralPath $StdoutPath).Length }
                if (Test-Path -LiteralPath $StderrPath) { $beat.stderr_bytes = (Get-Item -LiteralPath $StderrPath).Length }
                if ($OnHeartbeat) {
                    try { & $OnHeartbeat $beat }
                    catch {
                        $cause = $_.Exception.GetBaseException()
                        if ($cause -isnot [IO.IOException] -and $cause -isnot [UnauthorizedAccessException]) { throw }
                        # A locked status file must not terminate the owned child.
                        # Keep the last valid status; retry on the next heartbeat.
                        Write-Warning ('PRED heartbeat could not be saved; research continues: ' + $cause.Message)
                    }
                }
                $lastHeartbeat = $now
            }
            if (($now - $startedAt).TotalSeconds -gt $TimeoutSeconds) {
                # Stop only the exact child started by this invocation, plus its descendants.
                & "$env:SystemRoot/System32/taskkill.exe" /PID $process.Id /T /F | Out-Null
                if ($LASTEXITCODE -ne 0) { throw 'PRED timeout reached, but stopping its owned process tree failed; inspect the saved worker identity before recovery.' }
                if (-not $process.WaitForExit(5000)) { throw 'Owned PRED child did not exit after its timeout stop request.' }
                throw 'PRED child exceeded its execution timeout; output is preserved for recovery.'
            }
        } while (-not $process.WaitForExit(250))
        $null = $pumpOut.GetAwaiter().GetResult()
        $null = $pumpError.GetAwaiter().GetResult()
        return $process.ExitCode
    } finally {
        if ($awake) { [PredProcessStreams]::KeepAwake($false) }
        $process.Dispose()
    }
}
