$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$alignment = Join-Path $repo 'results\d4a_repair_alignment.csv'
$integrated = Join-Path $repo 'results\D4_integrated_report.md'
$d4b = Join-Path $repo 'results\D4b_residual_desync_results.md'
$protocol = Join-Path $repo 'source\D4_PROTOCOL.md'
$impl = Join-Path $repo 'source\d4b_residual_desync_gpu.py'
$numberAudit = Join-Path $repo 'repro\audit_claim_numbers.py'

$required = @($alignment, $integrated, $d4b, $protocol, $impl, $numberAudit)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required artifact: $path"
    }
}

$rows = Import-Csv -LiteralPath $alignment
if ($rows.Count -ne 8) {
    throw "Expected 8 formal D4a alignment rows, found $($rows.Count)"
}

$requiredColumns = @('split','variant','K','n','dyn_to_util_q90_err_mean','dyn_null_to_util_q90_err_mean','rank_to_util_q90_err_mean','dyn_closer_q90_rate')
$headers = ($rows[0].PSObject.Properties | ForEach-Object Name)
foreach ($column in $requiredColumns) {
    if ($headers -notcontains $column) {
        throw "Missing D4a column: $column"
    }
}

$reportText = Get-Content -Raw -LiteralPath $d4b
if ($reportText -notmatch '3000 configurations') {
    throw 'D4b report does not document the expected formal configuration count.'
}
if ($reportText -notmatch 'residual strength') {
    throw 'D4b report does not contain the strength-level summary.'
}

Write-Output "D4a rows: $($rows.Count)"
Write-Output "D4a splits: $((@($rows | Select-Object -ExpandProperty split -Unique)) -join ', ')"
Write-Output "D4a variants: $((@($rows | Select-Object -ExpandProperty variant -Unique)) -join ', ')"
Write-Output 'D4b formal raw CSV: not present in this repository; report-level aggregate is intentionally retained.'
python $numberAudit
if ($LASTEXITCODE -ne 0) {
    throw 'Claim-number audit failed.'
}
Write-Output 'Validation passed for the current evidence-bounded repository.'
