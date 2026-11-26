param(
  [Parameter(Mandatory=$true)][string]$SourceHost,
  [int]$SourcePort=1222,
  [Parameter(Mandatory=$true)][string]$SourceArchive,
  [string]$TargetHost='rmvision',
  [Parameter(Mandatory=$true)][string]$TargetArchive,
  [Parameter(Mandatory=$true)][string]$Receiver,
  [Parameter(Mandatory=$true)][long]$ArchiveBytes,
  [Parameter(Mandatory=$true)][string]$ArchiveSha256,
  [Parameter(Mandatory=$true)][string]$ReceiptPath
)
# SSH transport plumbing only. No scientific code/model executes on Windows.
# Password entry stays in SSH's inherited interactive console. Payloads remain
# in memory and SSH stdin; never print, log or persist payload/credentials here.
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $ReceiptPath) { throw 'New receipt path required; preserve earlier receipts.' }
foreach ($routingPath in @($SourceArchive,$TargetArchive,$Receiver)) {
  if ($routingPath -notmatch '^/[A-Za-z0-9_./-]+$') { throw 'Controlled absolute remote paths required.' }
}
if ($ArchiveSha256 -notmatch '^[a-f0-9]{64}$') { throw 'Complete SHA256 required.' }
$routingCommand="python3 $Receiver --out $TargetArchive --bytes $ArchiveBytes --sha256 $ArchiveSha256"
$routingStatusRaw=& ssh -T -o BatchMode=yes -o ConnectTimeout=15 $TargetHost ($routingCommand+' --status')
if ($LASTEXITCODE -ne 0) { throw 'Target status SSH failed.' }
$routingStatus=$routingStatusRaw | ConvertFrom-Json
if ($routingStatus.complete) {
  $routingStatus | ConvertTo-Json | Set-Content -LiteralPath $ReceiptPath
  Write-Output 'ALREADY_VERIFIED_COMPLETE';exit 0
}
$routingOffset=[long]$routingStatus.offset
$routingStart=$routingOffset
function Start-RoutingSsh([string[]]$Arguments,[bool]$InputPipe) {
  $routingInfo=[Diagnostics.ProcessStartInfo]::new()
  $routingInfo.FileName=(Get-Command ssh).Source
  $routingInfo.UseShellExecute=$false
  $routingInfo.RedirectStandardOutput=$true
  $routingInfo.RedirectStandardInput=$InputPipe
  $routingInfo.RedirectStandardError=$false
  foreach ($routingArg in $Arguments) { $routingInfo.ArgumentList.Add($routingArg) }
  $routingProcess=[Diagnostics.Process]::new();$routingProcess.StartInfo=$routingInfo
  [void]$routingProcess.Start();return $routingProcess
}
$routingSource=Start-RoutingSsh @('-T','-p',"$SourcePort",'-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
  $SourceHost,"tail -c +$($routingOffset+1) $SourceArchive") $false
$routingTarget=$null;$routingAcks=[Collections.Generic.List[object]]::new()
$routingBuffer=[byte[]]::new(1048576)
try {
  while ($routingOffset -lt $ArchiveBytes) {
    $routingWanted=[int][Math]::Min(1048576,$ArchiveBytes-$routingOffset);$routingFilled=0
    while ($routingFilled -lt $routingWanted) {
      $routingRead=$routingSource.StandardOutput.BaseStream.Read($routingBuffer,$routingFilled,$routingWanted-$routingFilled)
      if ($routingRead -eq 0) { throw 'Source SSH ended before committed archive boundary.' }
      $routingFilled+=$routingRead
    }
    # Open receiver only after source authentication and the first full frame.
    if ($null -eq $routingTarget) {
      $routingTarget=Start-RoutingSsh @('-T','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15',$TargetHost,$routingCommand) $true
    }
    $routingHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::Create().ComputeHash($routingBuffer,0,$routingFilled)).ToLowerInvariant()
    $routingFrame=[Convert]::ToBase64String($routingBuffer,0,$routingFilled)
    $routingTarget.StandardInput.WriteLine("$routingOffset|$routingHash|$routingFrame")
    $routingTarget.StandardInput.Flush();$routingFrame=$null
    $routingAckLine=$routingTarget.StandardOutput.ReadLine()
    if ($null -eq $routingAckLine) { throw 'Receiver SSH closed before frame acknowledgement.' }
    $routingAck=$routingAckLine | ConvertFrom-Json
    if ([long]$routingAck.offset -ne $routingOffset+$routingFilled -or $routingAck.chunk_sha256 -ne $routingHash) {
      throw 'Receiver acknowledgement differs.'
    }
    $routingOffset=[long]$routingAck.offset;$routingAcks.Add($routingAck)
    Write-Output "VERIFIED_BYTES $routingOffset"
  }
  $routingSource.WaitForExit()
  if ($routingSource.ExitCode -ne 0) { throw 'Source SSH exit failure.' }
  $routingTarget.StandardInput.WriteLine('END');$routingTarget.StandardInput.Close()
  $routingFinal=$routingTarget.StandardOutput.ReadLine() | ConvertFrom-Json
  $routingTarget.WaitForExit()
  if ($routingTarget.ExitCode -ne 0 -or -not $routingFinal.complete -or $routingFinal.archive_sha256 -ne $ArchiveSha256) {
    throw 'Receiver final verification failed.'
  }
  $routingReceipt=@{status='VERIFIED_MEMORY_ONLY_SSH_TRANSFER';archive_bytes=$ArchiveBytes;archive_sha256=$ArchiveSha256;
    resume_start=$routingStart;verified_offset=$routingOffset;chunks=$routingAcks;local_payload_files=0;credential_files=0}
  $routingReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath
  Write-Output 'VERIFIED_COMPLETE'
} finally {
  if (-not $routingSource.HasExited) { $routingSource.Kill() }
  if ($null -ne $routingTarget -and -not $routingTarget.HasExited) { $routingTarget.Kill() }
  [Array]::Clear($routingBuffer);$routingSource.Dispose()
  if ($null -ne $routingTarget) { $routingTarget.Dispose() }
}
