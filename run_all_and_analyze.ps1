# run_all_and_analyze.ps1
#
# Purpose:
# 1) Run full pipeline via main.py (current data layout)
# 2) Detect latest run tag from output files
# 3) Generate cluster_debug CSV for that run
# 4) Run facet quality analysis (latest strategy)
# 5) Copy analysis outputs into run-specific folder
#
# Notes:
# - No non-ASCII characters (avoid console encoding issues)
# - Assumes execution from project root

$ErrorActionPreference = "Stop"

# Ensure script runs from project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Optional: activate virtual environment
# . .\.venv\Scripts\Activate.ps1

Write-Host "=== [1/4] Running main pipeline (main.py) ==="
python main.py

Write-Host "=== [2/4] Searching latest run tag ==="
$clusterFiles = Get-ChildItem -Recurse -Path "output" -Filter "*_clauses_clustered_*.xlsx"

if (-not $clusterFiles) {
    Write-Error "No clustered XLSX files found under output/. Check main.py results."
    exit 1
}

# Extract run tags from filenames: *_clauses_clustered_<RUN_TAG>.xlsx
$runTags = @()
foreach ($f in $clusterFiles) {
    if ($f.Name -match "_clauses_clustered_(.+)\.xlsx$") {
        $runTags += $Matches[1]
    }
}

if (-not $runTags) {
    Write-Error "Failed to extract run tags from clustered filenames."
    exit 1
}

$latestRunTag = ($runTags | Sort-Object)[-1]
Write-Host ("Latest run tag detected: {0}" -f $latestRunTag)

$analysisDir = "output/analysis_$latestRunTag"
if (-not (Test-Path $analysisDir)) {
    New-Item -ItemType Directory -Path $analysisDir | Out-Null
}

Write-Host "=== [3/4] Generating cluster_debug CSV ==="
$inputGlob = "output/*/*_clauses_clustered_${latestRunTag}.xlsx"
$clusterDebugPath = "$analysisDir/cluster_debug_${latestRunTag}.csv"

python scripts/export_cluster_debug.py `
    --input_glob "$inputGlob" `
    --output "$clusterDebugPath"

Write-Host "=== [4/4] Running facet quality analysis (latest) ==="
python scripts/analyze_facet_quality.py --strategy latest

# Copy analysis outputs into run-specific directory
$bucketSamples = "output/bucket_example_samples.csv"
$facetStats    = "output/facet_bucket_stats.csv"
$facetCross    = "output/facet_vs_bucket_cross.csv"

if (Test-Path $bucketSamples) {
    Copy-Item $bucketSamples "$analysisDir/bucket_example_samples_${latestRunTag}.csv" -Force
}
if (Test-Path $facetStats) {
    Copy-Item $facetStats "$analysisDir/facet_bucket_stats_${latestRunTag}.csv" -Force
}
if (Test-Path $facetCross) {
    Copy-Item $facetCross "$analysisDir/facet_vs_bucket_cross_${latestRunTag}.csv" -Force
}

Write-Host "=== DONE ==="
Write-Host ("Run tag           : {0}" -f $latestRunTag)
Write-Host ("Cluster debug CSV : {0}" -f $clusterDebugPath)
Write-Host ("Analysis directory: {0}" -f $analysisDir)
