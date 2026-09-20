$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$alignment = Join-Path $repo 'results\d4a_repair_alignment.csv'
$integrated = Join-Path $repo 'results\D4_integrated_report.md'
$d4b = Join-Path $repo 'results\D4b_residual_desync_results.md'
 $d4bFormal = Join-Path $repo 'results\d4b_formal_gpu'
$protocol = Join-Path $repo 'source\D4_PROTOCOL.md'
$impl = Join-Path $repo 'source\d4b_residual_desync_gpu.py'
$numberAudit = Join-Path $repo 'repro\audit_claim_numbers.py'

$formalRequired = @(
    (Join-Path $d4bFormal 'curves.csv'),
    (Join-Path $d4bFormal 'cstar_summary.csv'),
    (Join-Path $d4bFormal 'd4b_strength_summary.csv'),
    (Join-Path $d4bFormal 'merge_manifest.json'),
    (Join-Path $d4bFormal 'PROVENANCE.md')
)
$required = @($alignment, $integrated, $d4b, $protocol, $impl, $numberAudit) + $formalRequired
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

$rawRows = @(Import-Csv -LiteralPath (Join-Path $d4bFormal 'curves.csv'))
$summaryRows = @(Import-Csv -LiteralPath (Join-Path $d4bFormal 'cstar_summary.csv'))
$groupRows = @(Import-Csv -LiteralPath (Join-Path $d4bFormal 'd4b_strength_summary.csv'))
if ($rawRows.Count -ne 3000) { throw "Expected 3000 formal D4b raw rows, found $($rawRows.Count)" }
if ($summaryRows.Count -ne 300) { throw "Expected 300 formal D4b seed summaries, found $($summaryRows.Count)" }
if ($groupRows.Count -ne 15) { throw "Expected 15 formal D4b strength/split groups, found $($groupRows.Count)" }

Write-Output "D4a rows: $($rows.Count)"
Write-Output "D4a splits: $((@($rows | Select-Object -ExpandProperty split -Unique)) -join ', ')"
Write-Output "D4a variants: $((@($rows | Select-Object -ExpandProperty variant -Unique)) -join ', ')"
Write-Output "D4b formal raw rows: $($rawRows.Count); seed summaries: $($summaryRows.Count); strength groups: $($groupRows.Count)"
python $numberAudit
if ($LASTEXITCODE -ne 0) {
    throw 'Claim-number audit failed.'
}
Write-Output 'Validation passed for the current evidence-bounded repository.'
