param(
    [string]$Distro = "Ubuntu-22.04"
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$identity
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    Write-Error "Please run this script in an Administrator PowerShell window."
}

Write-Host "Installing WSL2 and $Distro if they are not already available..."
wsl --install -d $Distro

Write-Host ""
Write-Host "If Windows asks for a reboot, reboot first."
Write-Host "After Ubuntu opens for the first time, create the Linux username/password."
Write-Host "Then run:"
Write-Host "  wsl -d $Distro -- bash /mnt/d/codex/elec_power_flow/hybrid_power_system_analysis/scripts/install_omnisci_heavydb_wsl.sh"
