param(
  [string]$Remote = "origin",
  [string]$BaseBranch = "main",
  [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

function W { param([string]$msg, [string]$color="Gray"); Write-Host $msg -ForegroundColor $color }
function OK { param([string]$msg); Write-Host ("  ✓ {0}" -f $msg) -ForegroundColor Green }
function NO { param([string]$msg); Write-Host ("  ✗ {0}" -f $msg) -ForegroundColor Red }
function HDR { param([string]$msg); Write-Host ("`n=== {0} ===" -f $msg) -ForegroundColor Cyan }

function Test-PatternsInFile {
  param([string]$Path, [string[]]$Patterns, [string]$Title)
  if (-not (Test-Path $Path)) { NO ("{0}: file not found ({1})" -f $Title, $Path); return $false }
  $ok = $true
  foreach ($p in $Patterns) {
    $hit = Select-String -Path $Path -Pattern $p -SimpleMatch -ErrorAction SilentlyContinue
    if ($null -eq $hit) { NO ("{0}: missing pattern: '{1}'" -f $Title, $p); $ok = $false } else { OK ("{0}: found '{1}'" -f $Title, $p) }
  }
  return $ok
}

try {
  if (-not (Test-Path ".git")) { throw "Run this script at the repository root ('.git' missing)." }

  HDR ("Git status vs {0}/{1}" -f $Remote, $BaseBranch)
  git fetch $Remote | Out-Null
  git status
  W ""
  W ("Changed files vs {0}/{1}:" -f $Remote, $BaseBranch) "DarkGray"
  git diff --name-status ("{0}/{1}..." -f $Remote, $BaseBranch) | Out-Host
  W ""
  W "Diff stat:" "DarkGray"
  git diff --stat ("{0}/{1}..." -f $Remote, $BaseBranch) | Out-Host

  # 1) config.py — COMMUNITY_* 기본값 존재
  HDR "config.py checks"
  $cfgOK = Test-PatternsInFile -Path "config.py" -Title "config" -Patterns @(
    'COMMUNITY_REL_TAU',
    'COMMUNITY_ALIAS_TAU',
    'COMMUNITY_BAN_MODE',
    'COMMUNITY_SUMMARY_MAX_SENTENCES',
    'COMMUNITY_RULES_PATH'
  )

  # 2) main.py — argparse 기본값이 config 사용 & community wiring
  HDR "main.py argparse defaults (from config)"
  $argOK = Test-PatternsInFile -Path "main.py" -Title "argparse→config" -Patterns @(
    'COMMUNITY_RULES_PATH',
    'COMMUNITY_REL_TAU',
    'COMMUNITY_ALIAS_TAU',
    'COMMUNITY_BAN_MODE',
    'COMMUNITY_SUMMARY_MAX_SENTENCES'
  )

  HDR "main.py community dual-score gate wiring"
  $gateOK = Test-PatternsInFile -Path "main.py" -Title "dual score/gate imports" -Patterns @(
    'dual_score_relevance',
    'keep_mask_gated',
    'build_alias_queries',
    'build_facet_queries'
  )

  HDR "main.py load/preprocess guards"
  $loadOK = Test-PatternsInFile -Path "main.py" -Title "load+preprocess" -Patterns @(
    'load_reviews(file_path)',
    'preprocess_reviews(df)',
    'df.columns.duplicated()',
    'REVIEW_ID_COL',
    'astype(str)',
    'review" not in df.columns'
  )

  HDR "main.py representative alias preference"
  $aliasRepOK = $false
  if (Select-String -Path "main.py" -Pattern 'alias_terms' -SimpleMatch -ErrorAction SilentlyContinue) {
    $aliasRepOK = $true
    OK "alias_terms parameter present"
  } else {
    NO "alias_terms parameter not found (run_full_pipeline signature or call-site)"
  }
  if (Select-String -Path "main.py" -Pattern 'sorted(.*alias_terms.*reverse=True)' -ErrorAction SilentlyContinue) {
    OK "representatives sorted to prefer alias-including sentences"
  } else {
    NO "representative alias-preference sort not found"
    $aliasRepOK = $false
  }

  # 3) 필수 파일 존재
  HDR "Required files presence"
  $reqFiles = @(
    'pipeline\__init__.py',
    'pipeline\summarizer_comm.py',
    'pipeline\community_loader.py',
    'pipeline\relevance_filter.py',
    'rules\community_rules.yml'
  )
  $presenceOK = $true
  foreach ($f in $reqFiles) {
    if (Test-Path $f) { OK $f } else { NO ("{0} (missing)" -f $f); $presenceOK = $false }
  }

  # 4) 런타임 임포트 테스트
  HDR "Runtime import test (Python)"
  $pyOK = $false
  if (-not (Test-Path $PythonExe)) {
    NO ("Python executable not found: {0}" -f $PythonExe)
  } else {
    $pycode = 'import pipeline.summarizer_comm, pipeline.community_loader, pipeline.relevance_filter; print("ok")'
    $out = & $PythonExe -c $pycode 2>&1
    if ( ($LASTEXITCODE -eq 0) -and ($out -match 'ok') ) { OK "Python import ok"; $pyOK = $true } else { NO ("Python import failed: {0}" -f $out) }
  }

  # 5) 요약 리포트
  HDR "Summary"
  $allOK = $cfgOK -and $argOK -and $gateOK -and $loadOK -and $aliasRepOK -and $presenceOK -and $pyOK
  if ($allOK) { OK "All spot checks passed. Repository looks consistent with the desired profile."; exit 0 }
  else { NO "Some checks failed. See sections above."; exit 1 }

} catch {
  NO $_.Exception.Message
  exit 2
}
