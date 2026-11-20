param(
    [Parameter(Mandatory=$true)][string]$Esc50Root,
    [string]$Python = "python",
    [string]$RunRoot = "runs/experiment"
)
$ErrorActionPreference = "Stop"
function Invoke-PythonChecked {
    param([string[]]$PythonArgs)
    & $Python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($PythonArgs -join ' ')" }
}
$resolvedEsc = (Resolve-Path -LiteralPath $Esc50Root).Path
Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath $RunRoot) { throw "RunRoot already exists. Choose a new directory." }
    Invoke-PythonChecked -PythonArgs @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print(torch.cuda.get_device_name(0))")
    Invoke-PythonChecked -PythonArgs @("download_assets.py", "--weights")
    $manifest = Join-Path $RunRoot "manifest.json"
    Invoke-PythonChecked -PythonArgs @("prepare.py", "--esc50", $resolvedEsc, "--out", $manifest)
    foreach ($mode in @("clean", "mixed")) {
        $run = Join-Path $RunRoot $mode
        $checkpoint = Join-Path $run "best.pt"
        $valOut = Join-Path $run "validation"
        $testOut = Join-Path $run "test"
        $thresholds = Join-Path $valOut "thresholds.json"
        Invoke-PythonChecked -PythonArgs @("train.py", "--mode", $mode, "--manifest", $manifest, "--out", $run)
        Invoke-PythonChecked -PythonArgs @("evaluate.py", "--checkpoint", $checkpoint, "--manifest", $manifest, "--split", "val", "--tune", "--out", $valOut)
        Invoke-PythonChecked -PythonArgs @("evaluate.py", "--checkpoint", $checkpoint, "--manifest", $manifest, "--split", "test", "--thresholds", $thresholds, "--out", $testOut)
    }
    Invoke-PythonChecked -PythonArgs @("plot_results.py", "--clean", (Join-Path $RunRoot "clean/test/report.json"), "--mixed", (Join-Path $RunRoot "mixed/test/report.json"), "--out", (Join-Path $RunRoot "figures"))
} finally {
    Pop-Location
}
