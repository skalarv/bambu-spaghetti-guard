# Windows service

Two viable paths. NSSM is the obvious answer for an always-on service. Task
Scheduler is fine if you only want the guard up while you're logged in.

## NSSM (recommended)

1. Download NSSM from <https://nssm.cc>.
2. From an elevated PowerShell:

   ```powershell
   nssm install SpaghettiGuard "G:\GitWorkSpace\BambuLab\bambu-spaghetti-guard\.venv\Scripts\spaghetti-guard.exe" run
   nssm set SpaghettiGuard AppDirectory "G:\GitWorkSpace\BambuLab\bambu-spaghetti-guard"
   nssm set SpaghettiGuard AppEnvironmentExtra ^
     "BAMBU_IP=192.168.1.50" ^
     "BAMBU_SERIAL=01P00A..." ^
     "BAMBU_ACCESS_CODE=xxxxxxxx"
   nssm set SpaghettiGuard Start SERVICE_AUTO_START
   nssm set SpaghettiGuard AppStdout "G:\GitWorkSpace\BambuLab\bambu-spaghetti-guard\guard.out.log"
   nssm set SpaghettiGuard AppStderr "G:\GitWorkSpace\BambuLab\bambu-spaghetti-guard\guard.err.log"
   nssm set SpaghettiGuard AppRotateFiles 1
   nssm set SpaghettiGuard AppRotateBytes 10485760
   nssm start SpaghettiGuard
   ```

3. `sc query SpaghettiGuard` to confirm RUNNING. `nssm restart`,
   `nssm remove SpaghettiGuard confirm` to manage it later.

Avoid putting `BAMBU_ACCESS_CODE` in `nssm edit` GUI text fields where it
ends up in plain view. The `nssm set ... AppEnvironmentExtra` form keeps it
in the registry under the service account; lock the registry path down with
ACLs if you're paranoid.

## Task Scheduler (lighter touch)

1. Open Task Scheduler → Create Task.
2. **General**: name `SpaghettiGuard`, "Run only when user is logged on" is
   fine for a workstation that's always on.
3. **Triggers**: At log on, of your user.
4. **Actions**: Start a program.
   - Program: `G:\GitWorkSpace\BambuLab\bambu-spaghetti-guard\.venv\Scripts\spaghetti-guard.exe`
   - Arguments: `run`
   - Start in: `G:\GitWorkSpace\BambuLab\bambu-spaghetti-guard`
5. **Conditions**: untick "Start the task only if the computer is on AC
   power" — desktops report battery sometimes.
6. **Settings**: tick "If the task is already running, do not start a new
   instance". Untick "Stop the task if it runs longer than ..."

Env vars in this mode come from your user profile — set them with
`setx BAMBU_IP 192.168.1.50` etc. once and log out / back in.

## Stopping the service

NSSM: `nssm stop SpaghettiGuard`.

Task Scheduler: right-click → End.
