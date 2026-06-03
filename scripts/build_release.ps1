# Builds the Windows x64 release zip.
# Output: easyonnx-windows-x64-v<VERSION>.zip in the repo root.
# Requires: CMake, Visual Studio Build Tools (cl.exe via Developer prompt).

$ErrorActionPreference = 'Stop'

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $RepoRoot

$rootCmake = Get-Content (Join-Path $RepoRoot 'CMakeLists.txt') -Raw
if ($rootCmake -notmatch 'project\(easy-onnx VERSION ([0-9.]+)') {
    throw "Failed to extract version from CMakeLists.txt"
}
$Version = $Matches[1]

$StageName = 'easyonnx-windows-x64'
$StageDir  = Join-Path $RepoRoot "stage\$StageName"
$BuildDir  = Join-Path $RepoRoot 'build-release'
$ZipName   = "$StageName-v$Version.zip"
$ZipPath   = Join-Path $RepoRoot $ZipName

Write-Host "==> Cleaning $BuildDir and $StageDir"
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }

Write-Host "==> Configuring (Release, x64, install prefix=$StageDir)"
cmake -B $BuildDir -S $RepoRoot -A x64 `
    -DCMAKE_INSTALL_PREFIX="$StageDir"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "==> Building"
cmake --build $BuildDir --config Release -j
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "==> Installing into staging directory"
cmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake install failed" }

Write-Host "==> Zipping $ZipName"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path $StageDir -DestinationPath $ZipPath

Write-Host ""
Write-Host "Done: $ZipPath"
Get-Item $ZipPath | Format-Table Name, Length
