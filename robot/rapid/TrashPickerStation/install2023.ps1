<#
.SYNOPSIS
    Builds and installs the TrashPicker Station add-in for RobotStudio 2023 (.NET Framework 4.8).

.DESCRIPTION
    The same add-in as install.ps1 installs for RobotStudio 2026, compiled from the same
    StationAddin.cs by rs2023\TrashPickerStation2023.csproj. Needs a .NET SDK (7 or newer) and the
    .NET Framework 4.8 targeting pack. Puts TrashPickerStation.dll and its .rsaddin in
    <packages>\TrashPickerStation-<version>\RobotStudio\Add-In, the layout of the robot-model
    packages there. RobotStudio scans packages only when it starts.

.PARAMETER RobotStudioBin
    RobotStudio 2023's Bin folder. Default: C:\Program Files (x86)\ABB\RobotStudio 2023\Bin.

.PARAMETER Packages
    The package folder RobotStudio 2023 scans. Default: C:\ProgramData\ABB\DistributionPackages,
    where its RobotWare and robot-model packages are (see
    %LOCALAPPDATA%\ABB\RobotStudio\ManifestCachev4.xml). %LOCALAPPDATA%\ABB\DistributionPackages2
    is RobotStudio 2026's: an Add-In folder there was not loaded by 2023.

.PARAMETER Uninstall
    Remove the 2023 add-in (the Add-In folder) instead.
#>
param(
    [string]$RobotStudioBin = 'C:\Program Files (x86)\ABB\RobotStudio 2023\Bin',
    [string]$Packages = (Join-Path $env:ProgramData 'ABB\DistributionPackages'),
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$version = ([xml](Get-Content (Join-Path $here 'manifest.xml'))).DistributionPackage.DisplayVersion
$target = Join-Path $Packages "TrashPickerStation-$version"
$addin = Join-Path $target 'RobotStudio\Add-In'

if (Get-Process RobotStudio -ErrorAction SilentlyContinue) {
    throw "RobotStudio is running and may hold the add-in open. Close it, then run this again."
}
if ($Uninstall) {
    if (Test-Path $addin) { Remove-Item -Recurse -Force $addin; Write-Host "Removed $addin" }
    return
}
if (-not (Test-Path (Join-Path $RobotStudioBin 'ABB.Robotics.RobotStudio.Stations.dll'))) {
    throw "No RobotStudio SDK assemblies in $RobotStudioBin. Pass -RobotStudioBin '<RobotStudio 2023>\Bin'."
}
& dotnet build (Join-Path $here 'rs2023\TrashPickerStation2023.csproj') -c Release -nologo "-p:RobotStudioBin=$RobotStudioBin"
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed" }
$built = Join-Path $here 'rs2023\bin\Release\net48'
New-Item -ItemType Directory -Force $addin | Out-Null
Copy-Item (Join-Path $built 'TrashPickerStation.dll'), (Join-Path $here 'TrashPickerStation.rsaddin') $addin -Force
Copy-Item (Join-Path $here 'manifest.xml') $target -Force
Write-Host "Installed to $addin"
Write-Host "Start RobotStudio 2023 and open the station. Its Output window should then say:"
Write-Host "  TrashPicker station: listening on 127.0.0.1:5003 ..."
