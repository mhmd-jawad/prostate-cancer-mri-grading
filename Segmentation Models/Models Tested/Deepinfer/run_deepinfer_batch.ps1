param(
    [string]$InputDir = (Join-Path $PSScriptRoot "nrrd_images"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "deepinfer_results_first25"),
    [int]$Count = 25,
    [switch]$UseGpu,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputDir)) {
    throw "Input directory not found: $InputDir"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$cases = Get-ChildItem -Path $InputDir -Filter "*.nrrd" |
    Sort-Object Name |
    Select-Object -First $Count

if (-not $cases) {
    throw "No .nrrd files found in $InputDir"
}

$inputMount = "${InputDir}:/in"
$outputMount = "${OutputDir}:/out"

foreach ($case in $cases) {
    $outputName = "{0}_seg.nrrd" -f $case.BaseName
    $outputPath = Join-Path $OutputDir $outputName

    if ($SkipExisting -and (Test-Path $outputPath)) {
        Write-Host "Skipping $($case.Name) because $outputName already exists."
        continue
    }

    $dockerArgs = @("run", "--rm", "-t")
    if ($UseGpu) {
        $dockerArgs += @("--gpus", "all")
    }
    $dockerArgs += @(
        "-v", $inputMount,
        "-v", $outputMount,
        "deepinfer/prostate",
        "--ModelName", "prostate-segmenter",
        "--Domain", "PROMISE12",
        "--InputVolume", "/in/$($case.Name)",
        "--OutputLabel", "/out/$outputName",
        "--ProcessingType", "Accurate",
        "--Inference", "Ensemble",
        "--verbose"
    )

    Write-Host "Processing $($case.Name) -> $outputName"
    & docker @dockerArgs

    if ($LASTEXITCODE -ne 0) {
        throw "DeepInfer failed for $($case.Name) with exit code $LASTEXITCODE"
    }
}

Write-Host "Completed processing $($cases.Count) case(s) into $OutputDir"
