# Windows service

Two viable paths. NSSM is the obvious answer for an always-on service. Task
Scheduler is fine if you only want the guard up while you're logged in.

`$INSTALL` below stands for your checkout directory (e.g.
`C:\opt\bambu-spaghetti-guard`).

## Secrets

`spaghetti-guard run` loads `secrets.local.txt` from its working directory
automatically (explicit environment variables win over the file). For a
service the simplest safe setup is:

1. Copy `secrets.local.txt.template` to `$INSTALL\secrets.local.txt` and fill it in.
2. Restrict it to the service account:

   ```powershell
   icacls "$INSTALL\secrets.local.txt" /inheritance:r /grant "$env:USERNAME:R" "SYSTEM:R"
   ```

Never pass `BAMBU_ACCESS_CODE` on a command line (`nssm set ...` arguments
land in PowerShell history and can surface in event logs) and never type it
into the `nssm edit` GUI.

## NSSM (recommended)

1. Download NSSM from <https://nssm.cc>.
2. From an elevated PowerShell:

   ```powershell
   nssm install SpaghettiGuard "$INSTALL\.venv\Scripts\spaghetti-guard.exe" run
   nssm set SpaghettiGuard AppDirectory "$INSTALL"
   nssm set SpaghettiGuard Start SERVICE_AUTO_START
   nssm set SpaghettiGuard AppStdout "$INSTALL\guard.out.log"
   nssm set SpaghettiGuard AppStderr "$INSTALL\guard.err.log"
   nssm set SpaghettiGuard AppRotateFiles 1
   nssm set SpaghettiGuard AppRotateBytes 10485760
   # Restart on any exit — the guard exits 3 on camera-reconnect exhaustion
   # and 4 on MQTT connect failure; both must bring it back.
   nssm set SpaghettiGuard AppExit Default Restart
   nssm set SpaghettiGuard AppRestartDelay 10000
   nssm start SpaghettiGuard
   ```

3. `sc query SpaghettiGuard` to confirm RUNNING. `nssm restart`,
   `nssm remove SpaghettiGuard confirm` to manage it later.

## Task Scheduler (lighter touch)

1. Open Task Scheduler → Create Task.
2. **General**: name `SpaghettiGuard`, "Run only when user is logged on" is
   fine for a workstation that's always on.
3. **Triggers**: At log on, of your user.
4. **Actions**: Start a program.
   - Program: `$INSTALL\.venv\Scripts\spaghetti-guard.exe`
   - Arguments: `run`
   - Start in: `$INSTALL`
5. **Conditions**: untick "Start the task only if the computer is on AC
   power" — desktops report battery sometimes.
6. **Settings**: tick "If the task is already running, do not start a new
   instance". Untick "Stop the task if it runs longer than ..."

Secrets come from `$INSTALL\secrets.local.txt` here too — no `setx` needed.

## Stopping the service

NSSM: `nssm stop SpaghettiGuard`.

Task Scheduler: right-click → End.
