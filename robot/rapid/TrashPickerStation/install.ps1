<#
.SYNOPSIS
    Installs the TrashPicker Station add-in for RobotStudio (see STATION_SETUP.md).

.DESCRIPTION
    Copies the prebuilt package in .\package to RobotStudio's package folder,
    %LOCALAPPDATA%\ABB\DistributionPackages2\TrashPickerStation-<version>, where RobotStudio 2026
    keeps its own packages (Name-Version folders). No .NET SDK is needed for that.
    With -Build, rebuilds the add-in first (needs the .NET 10 SDK) and refreshes .\package.
    RobotStudio has to be restarted afterwards; it scans packages only when it starts.

.PARAMETER Build
    Rebuild TrashPickerStation.dll from StationAddin.cs before installing.

.PARAMETER RobotStudioBin
    RobotStudio's Bin folder, for -Build. Default: found by Find-RobotStudioBin -- the development
    machine's folder if it exists, else a running RobotStudio.exe's folder, else the newest
    ABB\RobotStudio*\Bin under Program Files (x86) or Program Files.

.PARAMETER Uninstall
    Remove the installed package instead.
#>
param(
    [switch]$Build,
    [string]$RobotStudioBin = '',
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

# RobotStudio is not always where it was on the development machine: look where it runs or is
# installed. The SDK assemblies -Build compiles against are in its Bin folder.
function Find-RobotStudioBin {
    $sdk = 'ABB.Robotics.RobotStudio.Stations.dll'
    $candidates = @('C:\Program Files (x86)\Development\RobotStudio2026\Bin')
    $candidates += @(Get-Process RobotStudio -ErrorAction SilentlyContinue |
                     ForEach-Object { if ($_.Path) { Split-Path $_.Path } })
    foreach ($root in @(${env:ProgramFiles(x86)}, $env:ProgramFiles) | Where-Object { $_ }) {
        $candidates += @(Get-ChildItem (Join-Path $root 'ABB') -Directory -Filter 'RobotStudio*' -ErrorAction SilentlyContinue |
                         Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'Bin' })
    }
    foreach ($bin in $candidates) {
        if ($bin -and (Test-Path (Join-Path $bin $sdk))) { return $bin }
    }
    return $null
}
$package = Join-Path $here 'package'
# RobotStudio 2026 looks for an add-in in <package>\RobotStudio\Add-In-net10.0 (and the older
# Add-In / Add-In-net8.0 names); the add-in is built for net10.0.
$addinFolder = 'RobotStudio\Add-In-net10.0'
$version = ([xml](Get-Content (Join-Path $here 'manifest.xml'))).DistributionPackage.DisplayVersion
$packages = Join-Path $env:LOCALAPPDATA 'ABB\DistributionPackages2'
$target = Join-Path $packages "TrashPickerStation-$version"

# A running RobotStudio holds the installed DLL open, so a copy over it fails half way.
function Assert-RobotStudioClosed {
    if (Get-Process RobotStudio -ErrorAction SilentlyContinue) {
        throw "RobotStudio is running and holds the installed add-in open. Close it, then run this again."
    }
}

# Any version installed before, so RobotStudio never finds two add-ins on one port.
$installed = @(Get-ChildItem $packages -Directory -Filter 'TrashPickerStation-*' -ErrorAction SilentlyContinue)
if ($Uninstall) {
    Assert-RobotStudioClosed
    $installed | ForEach-Object { Remove-Item -Recurse -Force $_.FullName; Write-Host "Removed $($_.FullName)" }
    Write-Host "Restart RobotStudio."
    return
}

if ($Build) {
    if (-not $RobotStudioBin) {
        $RobotStudioBin = Find-RobotStudioBin
        if (-not $RobotStudioBin) {
            throw "RobotStudio's Bin folder was not found. Pass -RobotStudioBin '<RobotStudio>\Bin'."
        }
        Write-Host "RobotStudio Bin: $RobotStudioBin"
    }
    if (-not (Test-Path (Join-Path $RobotStudioBin 'ABB.Robotics.RobotStudio.Stations.dll'))) {
        throw "No RobotStudio SDK assemblies in $RobotStudioBin. Pass -RobotStudioBin '<RobotStudio>\Bin'."
    }
    & dotnet build (Join-Path $here 'TrashPickerStation.csproj') -c Release -nologo "-p:RobotStudioBin=$RobotStudioBin"
    if ($LASTEXITCODE -ne 0) { throw "dotnet build failed" }
    $built = Join-Path $here 'bin\Release\net10.0-windows'
    $dest = Join-Path $package $addinFolder
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $built 'TrashPickerStation.dll'), (Join-Path $here 'TrashPickerStation.rsaddin') $dest -Force
    Copy-Item (Join-Path $here 'manifest.xml') $package -Force
    Write-Host "Package refreshed in $package"
}

if (-not (Test-Path (Join-Path $package "$addinFolder\TrashPickerStation.dll"))) {
    throw "No prebuilt add-in in $package. Run .\install.ps1 -Build (needs the .NET 10 SDK)."
}
Assert-RobotStudioClosed
$installed | ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
New-Item -ItemType Directory -Force $target | Out-Null
Copy-Item (Join-Path $package '*') $target -Recurse -Force
Write-Host "Installed to $target"
Write-Host "Restart RobotStudio and open the station. Its Output window should then say:"
Write-Host "  TrashPicker station: listening on 127.0.0.1:5003 ..."
