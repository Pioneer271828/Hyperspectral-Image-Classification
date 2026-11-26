param(
  [Parameter(Mandatory=$true)][string]$SourceHost,
  [int]$SourcePort=1222,
  [ValidateSet(1048576,16777216)][int]$FrameBytes=1048576,
  [switch]$BoundedSourceChunks,
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
if ($FrameBytes -ne 1048576) { $routingCommand+=" --frame-bytes $FrameBytes" }
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
$routingSource=$null
if (-not $BoundedSourceChunks) {
  $routingSource=Start-RoutingSsh @('-T','-p',"$SourcePort",'-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
    $SourceHost,"tail -c +$($routingOffset+1) $SourceArchive") $false
}
$routingTarget=$null;$routingAcks=[Collections.Generic.List[object]]::new()
$routingBuffer=[byte[]]::new($FrameBytes)
$routingReportEvery=[int](67108864/$FrameBytes)
try {
  while ($routingOffset -lt $ArchiveBytes) {
    # Select the Int64 overload before narrowing the bounded one-MiB frame.
    # Otherwise PowerShell chooses Int32 from the first literal and rejects
    # archives above two GiB before reading any payload.
    $routingWanted=[int][Math]::Min([long]$FrameBytes,[long]($ArchiveBytes-$routingOffset));$routingFilled=0
    if ($BoundedSourceChunks) {
      # Each authenticated source command ends after exactly one bounded frame,
      # avoiding a blocked source stdout while the target durably acknowledges.
      $routingSource=Start-RoutingSsh @('-T','-p',"$SourcePort",'-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
        $SourceHost,"dd if=$SourceArchive iflag=skip_bytes,count_bytes skip=$routingOffset count=$routingWanted bs=1048576 status=none") $false
    }
    while ($routingFilled -lt $routingWanted) {
      $routingRead=$routingSource.StandardOutput.BaseStream.Read($routingBuffer,$routingFilled,$routingWanted-$routingFilled)
      if ($routingRead -eq 0) { throw 'Source SSH ended before committed archive boundary.' }
      $routingFilled+=$routingRead
    }
    if ($BoundedSourceChunks) {
      $routingSource.WaitForExit()
      if ($routingSource.ExitCode -ne 0) { throw 'Bounded source SSH exit failure.' }
      $routingSource.Dispose();$routingSource=$null
    }
    # Open receiver only after source authentication and the first full frame.
    if ($null -eq $routingTarget) {
      $routingReceiveCommand=$routingCommand
      if ($BoundedSourceChunks) { $routingReceiveCommand+=' --one-frame' }
      $routingTarget=Start-RoutingSsh @('-T','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15',$TargetHost,$routingReceiveCommand) $true
    }
    $routingHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::Create().ComputeHash($routingBuffer,0,$routingFilled)).ToLowerInvariant()
    $routingFrame=[Convert]::ToBase64String($routingBuffer,0,$routingFilled)
    # Write the ASCII frame as one native byte buffer. StreamWriter's small
    # character buffer creates thousands of synchronous pipe writes per frame
    # on Windows, which is unsuitable for multi-GiB recovery artifacts.
    $routingWire=[Text.Encoding]::ASCII.GetBytes("$routingOffset|$routingHash|$routingFrame`n")
    $routingTarget.StandardInput.BaseStream.Write($routingWire,0,$routingWire.Length)
    $routingTarget.StandardInput.BaseStream.Flush();$routingFrame=$null
    [Array]::Clear($routingWire);$routingWire=$null
    $routingAckLine=$routingTarget.StandardOutput.ReadLine()
    if ($null -eq $routingAckLine) { throw 'Receiver SSH closed before frame acknowledgement.' }
    $routingAck=$routingAckLine | ConvertFrom-Json
    if ([long]$routingAck.offset -ne $routingOffset+$routingFilled -or $routingAck.chunk_sha256 -ne $routingHash) {
      throw 'Receiver acknowledgement differs.'
    }
    $routingOffset=[long]$routingAck.offset;$routingAcks.Add($routingAck)
    if ($BoundedSourceChunks -and $routingOffset -lt $ArchiveBytes) {
      $routingTarget.StandardInput.Close();$routingTarget.WaitForExit()
      if ($routingTarget.ExitCode -ne 0) { throw 'Bounded target SSH exit failure.' }
      $routingTarget.Dispose();$routingTarget=$null
    }
    if ($routingAcks.Count % $routingReportEvery -eq 0 -or $routingOffset -eq $ArchiveBytes) { Write-Output "VERIFIED_BYTES $routingOffset" }
  }
  if ($null -ne $routingSource) {
    $routingSource.WaitForExit()
    if ($routingSource.ExitCode -ne 0) { throw 'Source SSH exit failure.' }
  }
  if (-not $BoundedSourceChunks) {
    $routingEnd=[Text.Encoding]::ASCII.GetBytes("END`n")
    $routingTarget.StandardInput.BaseStream.Write($routingEnd,0,$routingEnd.Length)
    $routingTarget.StandardInput.BaseStream.Flush()
  }
  $routingTarget.StandardInput.Close()
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
  if ($null -ne $routingSource -and -not $routingSource.HasExited) { $routingSource.Kill() }
  if ($null -ne $routingTarget -and -not $routingTarget.HasExited) { $routingTarget.Kill() }
  [Array]::Clear($routingBuffer)
  if ($null -ne $routingSource) { $routingSource.Dispose() }
  if ($null -ne $routingTarget) { $routingTarget.Dispose() }
}
