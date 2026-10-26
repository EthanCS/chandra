param(
    [switch]$BuildOnly,
    [switch]$Sponza,
    [switch]$Gallery,
    [switch]$Courtyard,
    [switch]$IntelSponza,
    [string]$ScenePath,
    [switch]$SmokeTest,
    [switch]$Benchmark,
    [switch]$Test,
    [switch]$StressTest,
    [switch]$Small,
    [switch]$FullHD,
    [switch]$ProfileGpu,
    [switch]$SingleFrame,
    [switch]$SplitSingleFrame,
    [switch]$Static,
    [ValidateRange(0, 2)][int]$View = 0,
    [ValidateRange(1, 16)][int]$PathDepth = 8,
    [ValidateRange(1, 64)][int]$Samples = 1,
    [ValidateRange(0, 256)][int]$MinProbeRays = 0,
    [ValidateRange(1, 100)][int]$Cubes = 1,
    [switch]$RigidOnly,
    [switch]$RigidRefit,
    [switch]$EmissiveDay,
    [switch]$EmissiveNight,
    [ValidateRange(0, 100)][float]$Emission = 4,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
if ($SplitSingleFrame -and $SingleFrame) { throw 'Choose -SplitSingleFrame or the experimental -SingleFrame, not both.' }
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot 'tools\prepare-luisa-frontend.ps1')
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ Build Tools are required.' }
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$cuda = Join-Path $PSScriptRoot '.local\cuda'
if (Test-Path (Join-Path $cuda 'bin\nvcc.exe')) { $env:CUDA_PATH = $cuda }
if (-not $env:CUDA_PATH) { throw 'CUDA Toolkit is required. See README.md.' }
$ninja = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
$env:PATH = "$ninja;$env:CUDA_PATH\bin;$env:PATH"
$env:CARGO_NET_GIT_FETCH_WITH_CLI = 'true'
$env:CARGO_HTTP_MULTIPLEXING = 'false'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '12'
$env:CMAKE_TOOLCHAIN_FILE = Join-Path $PSScriptRoot 'cmake\windows-compat.cmake'
$env:RUST_LOG = 'warn'
# Luisa's pinned build script assumes Cargo's pre-1.100 output directory layout.
# Use that Cargo driver with the newer compiler required by sefirot's SyncView.
$cargoDriver = & rustup which cargo --toolchain nightly-2026-03-02
if ($LASTEXITCODE -ne 0) { throw 'Run setup-lighting-lab.ps1 to install the Cargo driver.' }
$env:RUSTC = & rustup which rustc --toolchain nightly-2026-10-03
if ($LASTEXITCODE -ne 0) { throw 'Run setup-lighting-lab.ps1 to install the Rust compiler.' }
$buildArguments = @('build', '--locked', '--jobs', '12', '--bin', 'lighting_lab')
& $cargoDriver @buildArguments
if ($LASTEXITCODE -ne 0) { throw 'Chandra build failed.' }
if ($BuildOnly) { return }
if ($Test) {
    & $cargoDriver test --locked --jobs 12 --lib --test dynamic_geometry --test dynamic_history --test gi_optimizations --test emissive_lighting -- --test-threads=1
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
    return
}
$arguments = @()
if ($Sponza) { $arguments += '--sponza' }
if ($Gallery) { $arguments += '--gallery' }
if ($Courtyard) { $arguments += '--courtyard' }
if ($IntelSponza) { $arguments += '--intel-sponza' }
if ($ScenePath) { $arguments += @('--scene-path', $ScenePath) }
if ($SmokeTest) { $arguments += '--smoke-test' }
if ($Small) { $arguments += '--small' }
if ($Benchmark) { $arguments += '--benchmark' }
if ($StressTest) { $arguments += '--stress-test' }
if ($FullHD) { $arguments += '--1080p' }
if ($ProfileGpu) { $arguments += '--profile' }
if ($SingleFrame) { $arguments += @('--single-frame', '--path-depth', $PathDepth, '--samples', $Samples, '--min-probe-rays', $MinProbeRays) }
if ($SplitSingleFrame) { $arguments += '--split-single-frame' }
if ($Static) { $arguments += '--static' }
if ($View -ne 0) { $arguments += @('--view', $View) }
if ($Cubes -ne 1) { $arguments += @('--cubes', $Cubes) }
if ($RigidOnly) { $arguments += '--rigid-only' }
if ($RigidRefit) { $arguments += '--rigid-refit' }
if ($EmissiveDay) { $arguments += '--emissive-day' }
if ($EmissiveNight) { $arguments += '--emissive-night' }
if ($EmissiveDay -or $EmissiveNight) { $arguments += @('--emission', $Emission.ToString([cultureinfo]::InvariantCulture)) }
if ($Output) { $arguments += @('--output', $Output) }
& (Join-Path $PSScriptRoot 'target\debug\lighting_lab.exe') @arguments
if ($LASTEXITCODE -ne 0) { throw "Lighting Lab exited with code $LASTEXITCODE" }
