# systemd install

Tested on Debian 12 and Ubuntu 24.04. Adapt paths for your distro.

```bash
# Install
sudo useradd --system --home /opt/bambu-spaghetti-guard --shell /usr/sbin/nologin spaghetti
sudo install -d -o spaghetti -g spaghetti /opt/bambu-spaghetti-guard
sudo rsync -a --chown spaghetti:spaghetti ./ /opt/bambu-spaghetti-guard/

# Build venv
sudo -u spaghetti python3.11 -m venv /opt/bambu-spaghetti-guard/.venv
sudo -u spaghetti /opt/bambu-spaghetti-guard/.venv/bin/pip install -r /opt/bambu-spaghetti-guard/requirements.txt
sudo -u spaghetti /opt/bambu-spaghetti-guard/.venv/bin/pip install -e /opt/bambu-spaghetti-guard

# Live deps (only if you want detection on this host)
sudo -u spaghetti /opt/bambu-spaghetti-guard/.venv/bin/pip install ultralytics opencv-python torch

# Credentials
sudo tee /etc/spaghetti-guard.env >/dev/null <<EOF
BAMBU_IP=192.168.1.50
BAMBU_SERIAL=01P00A...
BAMBU_ACCESS_CODE=xxxxxxxx
EOF
sudo chmod 600 /etc/spaghetti-guard.env
sudo chown root:spaghetti /etc/spaghetti-guard.env

# Service
sudo cp /opt/bambu-spaghetti-guard/deploy/systemd/spaghetti-guard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now spaghetti-guard
sudo systemctl status spaghetti-guard
```

Logs: `journalctl -u spaghetti-guard -f`.

`/etc/spaghetti-guard.env` is the one file with secrets. Keep its mode at
`0640 root:spaghetti` so other users on the host can't read the access code.
