$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$ResultsDir = Join-Path $PSScriptRoot "results\robustness"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $ResultsDir |
    Out-Null


Write-Host ""
Write-Host "============================================================"
Write-Host "ROBUSTNESS OVERNIGHT RUN"
Write-Host "============================================================"
Write-Host ""


# ------------------------------------------------------------
# Check llama-server
# ------------------------------------------------------------

Write-Host "Checking llama-server..."

try {
    $health = Invoke-RestMethod `
        -Uri "http://127.0.0.1:8080/health" `
        -TimeoutSec 10
}
catch {
    Write-Host ""
    Write-Host "ERROR: llama-server is not reachable."
    Write-Host "Keep llama-server running on port 8080 before starting."
    exit 1
}

if ($health.status -ne "ok") {
    Write-Host "ERROR: unexpected llama-server status:"
    $health
    exit 1
}

Write-Host "llama-server: PASS"
Write-Host ""


function Run-Robustness {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Csv,
        [string]$Log
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "STARTING $Name"
    Write-Host "============================================================"
    Write-Host ""

    & python $Script 2>&1 |
        Tee-Object -FilePath $Log

    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "$Name failed with Python exit code $exitCode"
    }

    if (-not (Test-Path $Csv)) {
        throw "$Name finished but expected CSV was not created: $Csv"
    }

    $rows = @(Import-Csv $Csv)

    $count = $rows.Count

    $valid = @(
        $rows |
        Where-Object {
            $_.valid_format -eq "True"
        }
    ).Count

    $invalid = $count - $valid

    Write-Host ""
    Write-Host "$Name summary:"
    Write-Host "  rows:    $count / 840"
    Write-Host "  valid:   $valid"
    Write-Host "  invalid: $invalid"

    if ($count -ne 840) {
        throw "$Name ended with only $count / 840 observations."
    }

    Write-Host ""
    Write-Host "$Name COMPLETE"
}


# ------------------------------------------------------------
# R1
# Same frozen prompts, max_tokens=12
# Existing first 4 observations are automatically preserved.
# ------------------------------------------------------------

Run-Robustness `
    -Name "R1 — original prompt, max_tokens=12" `
    -Script "run_robustness_r1.py" `
    -Csv "results\robustness\r1_max12_raw.csv" `
    -Log "results\robustness\r1.log"


# ------------------------------------------------------------
# R2
# Strict output-format prompt, max_tokens=12
# ------------------------------------------------------------

Run-Robustness `
    -Name "R2 — strict-format prompt, max_tokens=12" `
    -Script "run_robustness_r2.py" `
    -Csv "results\robustness\r2_strict_prompt_raw.csv" `
    -Log "results\robustness\r2.log"


# ------------------------------------------------------------
# R3
# Semantically equivalent paraphrase, max_tokens=12
# ------------------------------------------------------------

Run-Robustness `
    -Name "R3 — paraphrased prompt, max_tokens=12" `
    -Script "run_robustness_r3.py" `
    -Csv "results\robustness\r3_paraphrase_raw.csv" `
    -Log "results\robustness\r3.log"


Write-Host ""
Write-Host "============================================================"
Write-Host "ALL ROBUSTNESS RUNS COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Expected outputs:"
Write-Host "  results\robustness\r1_max12_raw.csv"
Write-Host "  results\robustness\r2_strict_prompt_raw.csv"
Write-Host "  results\robustness\r3_paraphrase_raw.csv"
Write-Host ""