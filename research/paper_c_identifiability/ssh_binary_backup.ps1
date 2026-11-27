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
# Native SSH stream plumbing only; no scientific code or payload file on Windows.
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $ReceiptPath){throw 'New receipt path required'}
foreach($routingPath in @($SourceArchive,$TargetArchive,$Receiver)){
  if($routingPath -notmatch '^/[A-Za-z0-9_./-]+$'){throw 'Controlled absolute remote paths required'}
}
if($ArchiveBytes -le 0 -or $ArchiveSha256 -notmatch '^[a-f0-9]{64}$'){throw 'Complete archive size/hash required'}
$routingReceive="python3 $Receiver --out $TargetArchive --bytes $ArchiveBytes --sha256 $ArchiveSha256"
$routingStatusRaw=& ssh -T -o BatchMode=yes -o ConnectTimeout=6 $TargetHost ($routingReceive+' --status')
if($LASTEXITCODE -ne 0){throw 'Target resume status failed; no source stream opened'}
$routingStatus=$routingStatusRaw | ConvertFrom-Json
if($routingStatus.complete){$routingStatus | ConvertTo-Json | Set-Content -LiteralPath $ReceiptPath;Write-Output 'ALREADY_VERIFIED_COMPLETE';exit 0}
$routingOffset=[long]$routingStatus.offset
function Start-RoutingBinarySsh([string[]]$Arguments,[bool]$InputPipe){
  $routingInfo=[Diagnostics.ProcessStartInfo]::new();$routingInfo.FileName=(Get-Command ssh).Source
  $routingInfo.UseShellExecute=$false;$routingInfo.RedirectStandardOutput=$true
  $routingInfo.RedirectStandardInput=$InputPipe;$routingInfo.RedirectStandardError=$false
  foreach($routingArg in $Arguments){$routingInfo.ArgumentList.Add($routingArg)}
  $routingProcess=[Diagnostics.Process]::new();$routingProcess.StartInfo=$routingInfo
  [void]$routingProcess.Start();return $routingProcess
}
$routingSource=$null;$routingTarget=$null;$routingFirst=[byte[]]::new(65536)
try{
  $routingSourceCommand='test "$(stat -c %s {0})" = {1} && test "$(sha256sum {0} | cut -d" " -f1)" = {2} && tail -c +{3} {0}' -f $SourceArchive,$ArchiveBytes,$ArchiveSha256,($routingOffset+1)
  $routingSource=Start-RoutingBinarySsh @('-T','-p',"$SourcePort",'-o','ConnectTimeout=10','-o','ServerAliveInterval=15',
    '-o','ServerAliveCountMax=4','-o','IPQoS=throughput',$SourceHost,$routingSourceCommand) $false
  # Authenticate and obtain source bytes before opening the target writer.
  $routingRead=$routingSource.StandardOutput.BaseStream.Read($routingFirst,0,$routingFirst.Length)
  if($routingRead -eq 0){throw 'Source rejected size/hash or ended before first bytes'}
  $routingTarget=Start-RoutingBinarySsh @('-T','-o','BatchMode=yes','-o','ConnectTimeout=6','-o','ServerAliveInterval=15',
    '-o','ServerAliveCountMax=4','-o','IPQoS=throughput',$TargetHost,$routingReceive) $true
  # Drain metadata stdout concurrently, otherwise long copies can fill its pipe.
  $routingMetadataTask=$routingTarget.StandardOutput.ReadToEndAsync()
  $routingTarget.StandardInput.BaseStream.Write($routingFirst,0,$routingRead)
  $routingCopyTask=$routingSource.StandardOutput.BaseStream.CopyToAsync($routingTarget.StandardInput.BaseStream,65536)
  $routingCopyTask.GetAwaiter().GetResult()
  $routingSource.WaitForExit()
  if($routingSource.ExitCode -ne 0){throw 'Source SSH exit failure; inspect target committed prefix before retry'}
  $routingTarget.StandardInput.BaseStream.Flush();$routingTarget.StandardInput.Close();$routingTarget.WaitForExit()
  $routingLines=$routingMetadataTask.GetAwaiter().GetResult().Trim().Split("`n")
  $routingFinal=$routingLines[-1] | ConvertFrom-Json
  if($routingTarget.ExitCode -ne 0 -or -not $routingFinal.complete -or [long]$routingFinal.offset -ne $ArchiveBytes -or $routingFinal.archive_sha256 -ne $ArchiveSha256){
    throw 'Target whole-archive size/SHA publication failed'
  }
  @{status='VERIFIED_NATIVE_BINARY_SSH_STREAM';resume_start=$routingOffset;archive_bytes=$ArchiveBytes;archive_sha256=$ArchiveSha256;
    final=$routingFinal;local_payload_files=0;credential_files=0;new_optimizer_updates=0} |
      ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath
  Write-Output 'VERIFIED_COMPLETE'
}finally{
  foreach($routingProcess in @($routingSource,$routingTarget)){
    if($null -ne $routingProcess){if(-not $routingProcess.HasExited){$routingProcess.Kill()};$routingProcess.Dispose()}
  }
  [Array]::Clear($routingFirst)
}
