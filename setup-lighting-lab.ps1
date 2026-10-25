# Prepare the optional project-local CUDA SDK without changing system settings.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$sefirot = Join-Path (Split-Path $PSScriptRoot -Parent) 'sefirot'
if (-not (Test-Path (Join-Path $sefirot 'keter\Cargo.toml'))) {
    if (Test-Path $sefirot) { throw "Existing directory is not the expected dependency: $sefirot" }
    & git clone https://github.com/entropylost/sefirot $sefirot
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone sefirot.' }
    & git -C $sefirot checkout --detach 590f48f4d6cb329fd0833525348b621d43f4fd21
    if ($LASTEXITCODE -ne 0) { throw 'Could not select the tested dependency version.' }
}
$revision = & git -C $sefirot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $revision -ne '590f48f4d6cb329fd0833525348b621d43f4fd21') {
    throw 'Existing ../sefirot is not the tested revision. Keep it intact and use a separate checkout with the pinned dependency.'
}
New-Item -ItemType Directory -Path '.local\downloads', '.local\cuda' -Force | Out-Null
$manifest = Invoke-RestMethod 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json'
foreach ($part in @('cuda_cudart', 'cuda_nvcc', 'cuda_nvrtc', 'cuda_nvtx')) {
    $entry = $manifest.$part.'windows-x86_64'
    $archive = Join-Path $PSScriptRoot ".local\downloads\$part.zip"
    $valid = (Test-Path $archive) -and ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -eq $entry.sha256)
    if (-not $valid) {
        & curl.exe -L --fail --silent --show-error "https://developer.download.nvidia.com/compute/cuda/redist/$($entry.relative_path)" -o $archive
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $part" }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Checksum mismatch: $part"
    }
    & tar -xf $archive -C '.local/cuda' --strip-components=1
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $part" }
    Write-Host "Ready: $part"
}
& rustup toolchain install nightly-2026-03-02 --profile minimal
if ($LASTEXITCODE -ne 0) { throw 'Cargo driver setup failed.' }
& rustup toolchain install nightly-2026-10-03 --profile minimal
if ($LASTEXITCODE -ne 0) { throw 'Rust toolchain setup failed.' }
& (Join-Path $PSScriptRoot 'tools\prepare-luisa-frontend.ps1')
Write-Host 'Ready. Run .\run-lighting-lab.ps1 to build and open the viewer.'
