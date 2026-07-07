#requires -Version 7.0
<#
.SYNOPSIS
    Task runner for the Bambu spaghetti guard on Windows.

.DESCRIPTION
    Targets:
        setup      - create venv + install pinned deps + editable install
        test       - run unit + integration tests
        lint       - basic byte-compile check (full linters are deferred)
        replay     - replay a clip; pass clip path after target
        run-dry    - live pipeline, log payloads instead of publishing
        run-live   - live pipeline, real publish
        train      - fine-tune YOLO (requires CUDA torch + ultralytics)
        validate   - validate weights + emit summary

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 test
    .\tasks.ps1 replay verification\fixtures\my-clip
    .\tasks.ps1 run-dry
#>

param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('setup', 'test', 'lint', 'replay', 'run-dry', 'run-live', 'train', 'validate')]
    [string]$Target,
    # NOTE: not named $Args — that shadows PowerShell's automatic variable and
    # silently swallows arguments.
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$cli = Join-Path $venv 'Scripts\spaghetti-guard.exe'

function Invoke-WithVenv {
    param([string[]]$Cmd)
    if (-not (Test-Path $python)) {
        Write-Error "venv not found. Run '.\tasks.ps1 setup' first."
    }
    & $Cmd[0] @($Cmd[1..($Cmd.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        Write-Error "command failed with exit code $LASTEXITCODE"
    }
}

switch ($Target) {
    'setup' {
        if (-not (Test-Path $venv)) {
            & py -3.11 -m venv $venv
            if ($LASTEXITCODE -ne 0) { Write-Error 'py -3.11 not available' }
        }
        & $python -m pip install --upgrade pip
        & $python -m pip install -r requirements.txt
        & $python -m pip install -e .
        Write-Host "`nSetup complete. Activate with: .\.venv\Scripts\Activate.ps1"
    }
    'test' {
        Invoke-WithVenv @($python, '-m', 'pytest', '-q')
    }
    'lint' {
        # Byte-compile every .py — catches syntax errors without dragging in
        # a full linter dep. Add ruff / mypy here once they're pinned.
        Invoke-WithVenv @($python, '-m', 'compileall', '-q', 'src', 'verification', 'tests', 'training')
    }
    'replay' {
        if ($Rest.Length -lt 1) {
            Write-Error 'usage: .\tasks.ps1 replay <clip-path> [extra args...]'
        }
        Invoke-WithVenv (@($cli, 'replay') + $Rest)
    }
    'run-dry' {
        Invoke-WithVenv (@($cli, 'run', '--dry-run') + $Rest)
    }
    'run-live' {
        Write-Host 'Live mode — guard will publish real stop/pause commands.'
        Invoke-WithVenv (@($cli, 'run') + $Rest)
    }
    'train' {
        if ($Rest.Length -lt 1) {
            Write-Error 'usage: .\tasks.ps1 train --data training/data/data.yaml [more args]'
        }
        Invoke-WithVenv (@($python, 'training/train.py') + $Rest)
    }
    'validate' {
        if ($Rest.Length -lt 1) {
            Write-Error 'usage: .\tasks.ps1 validate --weights X --data Y'
        }
        Invoke-WithVenv (@($python, 'training/validate.py') + $Rest)
    }
}
