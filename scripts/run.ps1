$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install the project first. See README.md.'
}
# This is the interactive GUI. pythonw avoids a terminal; internal helpers use CREATE_NO_WINDOW.
Start-Process -FilePath $python -ArgumentList @('-m', 'jev_launcher') -WorkingDirectory $projectRoot -WindowStyle Normal
