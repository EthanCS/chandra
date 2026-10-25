# Generate a project-local frontend patch; never modify the shared Cargo cache.
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$pin = 'd270b5296f600a3147efedc2823766ff989d1b9e'
$url = 'https://github.com/entropylost/luisa-compute-rs'
$target = Join-Path $workspace '.local\luisa_compute'
$patch = Join-Path $PSScriptRoot 'luisa-accel-transform.rs'
$stamp = "$pin/$((Get-FileHash -LiteralPath $patch).Hash)/$((Get-FileHash -LiteralPath $PSCommandPath).Hash)"
$stampFile = Join-Path $target 'chandra-patch.txt'
if ((Test-Path $stampFile) -and (Get-Content -LiteralPath $stampFile -Raw).Trim() -eq $stamp) { return }
if ((Test-Path $target) -and -not (Test-Path $stampFile)) {
    throw "Refusing to overwrite an unrecognized local dependency: $target"
}
$cargoCache = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$sourceRoot = $null
foreach ($repo in Get-ChildItem -LiteralPath (Join-Path $cargoCache 'git\checkouts') -Directory -Filter 'luisa-compute-rs-*' -ErrorAction SilentlyContinue) {
    foreach ($checkout in Get-ChildItem -LiteralPath $repo.FullName -Directory -Filter 'd270b52*') {
        $revision = & git -C $checkout.FullName rev-parse HEAD
        if ($revision -eq $pin) { $sourceRoot = $checkout.FullName; break }
    }
    if ($sourceRoot) { break }
}
if (-not $sourceRoot) {
    $clone = Join-Path $workspace '.local\luisa-frontend-upstream'
    New-Item -ItemType Directory -Path (Join-Path $workspace '.local') -Force | Out-Null
    if (-not (Test-Path $clone)) {
        & git clone --filter=blob:none --no-checkout $url $clone
        if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned Luisa frontend.' }
    }
    & git -C $clone fetch origin $pin --depth 1
    if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned Luisa revision.' }
    $archive = Join-Path $workspace '.local\luisa-frontend.tar'
    & git -C $clone archive --format=tar --output=$archive $pin luisa_compute LICENSE
    if ($LASTEXITCODE -ne 0) { throw 'Could not export the pinned frontend.' }
    $sourceRoot = Join-Path $workspace '.local\luisa-frontend-source'
    New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null
    & tar -xf $archive -C $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not extract the pinned frontend.' }
}
$source = Join-Path $sourceRoot 'luisa_compute'
New-Item -ItemType Directory -Path $target -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $source 'src') -Destination $target -Recurse -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot 'LICENSE') -Destination $target -Force
$manifest = Get-Content -LiteralPath (Join-Path $source 'Cargo.toml') -Raw
# Keep the original Git source identity so Cargo.lock continues to pin every
# backend/IR/derive dependency to the same tested revision.
$manifest = [regex]::Replace($manifest, '(?m)^(luisa_compute\w*) = \{ path = "[^"]+", version = "([^"]+)" \}', '$1 = { git = "https://github.com/entropylost/luisa-compute-rs", branch = "main", version = "$2" }')
[IO.File]::WriteAllText((Join-Path $target 'Cargo.toml'), $manifest)
$rtxPath = Join-Path $target 'src\rtx.rs'
$rtx = (Get-Content -LiteralPath $rtxPath -Raw).Replace("`r`n", "`n")
$anchor = '    pub fn set_mesh('
if (-not $rtx.Contains($anchor)) { throw 'Pinned frontend patch context changed.' }
$rtx = $rtx.Replace($anchor, (Get-Content -LiteralPath $patch -Raw) + "`n" + $anchor)
$oldFlags = "        let mut flags = api::AccelBuildModificationFlags::PRIMITIVE;`n        dbg!(flags);"
if (-not $rtx.Contains($oldFlags)) { throw 'Pinned transform-flag patch context changed.' }
$rtx = $rtx.Replace($oldFlags, '        let mut flags = api::AccelBuildModificationFlags::PRIMITIVE | api::AccelBuildModificationFlags::TRANSFORM;')
[IO.File]::WriteAllText($rtxPath, $rtx)
[IO.File]::WriteAllText($stampFile, $stamp)
Write-Host 'Prepared project-local Luisa instance-transform support.'
