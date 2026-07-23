$ErrorActionPreference = 'Continue'
$mq = if ($env:MQIDS_PYTHON) { $env:MQIDS_PYTHON } else { 'python' }
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$common = @('--window-length','64','--moirai-layer','12','--projector','direct','--vocab-loss-weight','0','--device','cuda')
$runs = @(
  @{ name='prompt_minimal_l64_l12_direct_seed2027_clean_20260720'; seed='2027'; variant='minimal' },
  @{ name='prompt_minimal_l64_l12_direct_seed2028_20260720'; seed='2028'; variant='minimal' },
  @{ name='prompt_generic_l64_l12_direct_seed2027_20260720'; seed='2027'; variant='generic' },
  @{ name='prompt_generic_l64_l12_direct_seed2028_20260720'; seed='2028'; variant='generic' },
  @{ name='prompt_wrong_process_l64_l12_direct_seed2027_20260720'; seed='2027'; variant='wrong_process' },
  @{ name='prompt_wrong_process_l64_l12_direct_seed2028_20260720'; seed='2028'; variant='wrong_process' }
)

$log = Join-Path $root 'outputs\_prompt_counterfactual_batch.log'
'=== prompt counterfactual batch start ' + (Get-Date -Format 'o') | Out-File -FilePath $log -Encoding utf8

foreach ($r in $runs) {
  $start = Get-Date
  $line = '[' + $start.ToString('HH:mm:ss') + '] START ' + $r.name
  Write-Host $line -ForegroundColor Cyan
  $line | Out-File -FilePath $log -Append -Encoding utf8
  & $mq scripts/train.py --run-name $r.name --seed $r.seed --prompt-variant $r.variant @common *>&1 | Tee-Object -FilePath (Join-Path $root 'outputs\_batch_console.log') -Append | Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
  $end = Get-Date
  $elapsed = ($end - $start).TotalSeconds
  $done = '[' + $end.ToString('HH:mm:ss') + '] DONE ' + $r.name + ' elapsed=' + ([math]::Round($elapsed,1)) + 's'
  Write-Host $done -ForegroundColor Green
  $done | Out-File -FilePath $log -Append -Encoding utf8
}

'=== prompt counterfactual batch end ' + (Get-Date -Format 'o') | Out-File -FilePath $log -Append -Encoding utf8
Write-Host 'ALL DONE' -ForegroundColor Green
