param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,
    [Parameter(Mandatory = $true)]
    [string]$SourceDataDir
)

$ErrorActionPreference = "Stop"
$package = [System.IO.Path]::GetFullPath($PackageDir)
$sourceData = [System.IO.Path]::GetFullPath($SourceDataDir)
if (-not (Test-Path -LiteralPath $package -PathType Container)) {
    throw "Package directory does not exist: $package"
}
if (-not (Test-Path -LiteralPath $sourceData -PathType Container)) {
    throw "Source-data staging directory does not exist: $sourceData"
}

$overleafZip = Join-Path $package "CMPB_Overleaf_LaTeX_Source.zip"
$sourceZip = Join-Path $package "CMPB_supplementary_source_data.zip"
$completeZip = Join-Path $package "CMPB_complete_submission_package.zip"
$checksumPath = Join-Path $package "SHA256SUMS_CMPB.txt"

$rootTex = @(
    "mkg_cmpb.tex",
    "mkg_cmpb_supplement.tex",
    "mkg_cmpb_title_page.tex",
    "cover_letter_cmpb.tex"
)
$overleafNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($name in $rootTex) {
    [void]$overleafNames.Add($name)
}
foreach ($name in @(
    "mkg_cmpb.bib",
    "mkg_cmpb.bbl",
    "mkg_cmpb_supplement.bbl",
    "elsarticle.cls",
    "elsarticle-num.bst",
    "README_OVERLEAF.md"
)) {
    [void]$overleafNames.Add($name)
}

$figurePattern = '\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
foreach ($texName in $rootTex) {
    $texPath = Join-Path $package $texName
    if (-not (Test-Path -LiteralPath $texPath -PathType Leaf)) {
        throw "Missing required root source: $texPath"
    }
    $text = Get-Content -LiteralPath $texPath -Raw -Encoding UTF8
    foreach ($match in [regex]::Matches($text, $figurePattern)) {
        [void]$overleafNames.Add($match.Groups[1].Value)
    }
}

$overleafStage = Join-Path ([System.IO.Path]::GetTempPath()) (
    "MKG_CMPB_Overleaf_" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $overleafStage | Out-Null
foreach ($name in ($overleafNames | Sort-Object)) {
    $source = Join-Path $package $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Referenced Overleaf file is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $overleafStage $name)
}
Compress-Archive -Path (Join-Path $overleafStage "*") -DestinationPath $overleafZip -Force

$sourceFiles = Get-ChildItem -LiteralPath $sourceData -Recurse -File |
    Where-Object {
        $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache)[\\/]' -and
        $_.Extension -ne ".pyc"
    }
$sourceStage = Join-Path ([System.IO.Path]::GetTempPath()) (
    "MKG_CMPB_SourceData_" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $sourceStage | Out-Null
foreach ($file in $sourceFiles) {
    $relative = $file.FullName.Substring($sourceData.TrimEnd("\").Length + 1)
    $destination = Join-Path $sourceStage $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force |
        Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
}
Compress-Archive -Path (Join-Path $sourceStage "*") -DestinationPath $sourceZip -Force

$excludedExtensions = @(
    ".aux", ".log", ".blg", ".fdb_latexmk", ".fls", ".spl", ".synctex.gz"
)
$excludedNames = @(
    [System.IO.Path]::GetFileName($completeZip),
    [System.IO.Path]::GetFileName($checksumPath)
)
$completeFiles = Get-ChildItem -LiteralPath $package -File |
    Where-Object {
        $excludedNames -notcontains $_.Name -and
        $excludedExtensions -notcontains $_.Extension -and
        $_.Name -notmatch '\.synctex\.gz$'
    }
$completeStage = Join-Path ([System.IO.Path]::GetTempPath()) (
    "MKG_CMPB_Complete_" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $completeStage | Out-Null
foreach ($file in $completeFiles) {
    Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $completeStage $file.Name)
}
Compress-Archive -Path (Join-Path $completeStage "*") -DestinationPath $completeZip -Force

$hashFiles = Get-ChildItem -LiteralPath $package -File |
    Where-Object {
        $_.Name -ne [System.IO.Path]::GetFileName($checksumPath) -and
        $excludedExtensions -notcontains $_.Extension -and
        $_.Name -notmatch '\.synctex\.gz$'
    } |
    Sort-Object Name
$hashLines = foreach ($file in $hashFiles) {
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($file.Name)"
}
[System.IO.File]::WriteAllLines($checksumPath, $hashLines, [System.Text.UTF8Encoding]::new($false))

Write-Output "Overleaf files: $($overleafNames.Count)"
Write-Output "Source-data files: $($sourceFiles.Count)"
Write-Output "Complete-package files: $($completeFiles.Count)"
Write-Output "Checksums: $($hashLines.Count)"
