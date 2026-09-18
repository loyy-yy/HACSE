param(
    [string]$CondaEnv = "visa_ad",
    [switch]$SkipMissing
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$argsList = @("$Here\build_manifests.py")
if ($SkipMissing) { $argsList += "--skip-missing" }
& conda run -n $CondaEnv python @argsList
if ($LASTEXITCODE -ne 0) { throw "Manifest build failed with exit code $LASTEXITCODE" }
& conda run -n $CondaEnv python "$Here\tests\test_smoke.py"
if ($LASTEXITCODE -ne 0) { throw "Smoke tests failed with exit code $LASTEXITCODE" }

$Canonical = "$Here\..\stage6\outputs\stage6_v11_canonical_export\v11_canonical_predictions_all.csv"
if (Test-Path $Canonical) {
    & conda run -n $CondaEnv python "$Here\import_stage6_canonical_predictions.py"
    if ($LASTEXITCODE -ne 0) { throw "Stage6 prediction import failed with exit code $LASTEXITCODE" }
    & conda run -n $CondaEnv python "$Here\evaluate_predictions.py" `
        --manifest "$Here\outputs\manifests\mvtec_loco_manifest.csv" `
        --predictions "$Here\outputs\loco_canonical_predictions.csv" `
        --reference-method ODRC `
        --bootstrap 100 `
        --out-dir "$Here\outputs\loco_canonical_check"
    if ($LASTEXITCODE -ne 0) { throw "Canonical evaluation failed with exit code $LASTEXITCODE" }
}

Write-Host "Stage7 manifests, smoke tests, and canonical LOCO reproduction completed."
Write-Host "Next: re-extract common-schema V/O/R features, validate them, then run run_tabular_benchmark.py."
