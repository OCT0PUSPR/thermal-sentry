#!/usr/bin/env bash
#
# install_pi.sh -- one-shot Raspberry Pi setup for thermal-sentry.
#
# What it does:
#   1. Enables the I2C interface (and bumps the bus clock to 800 kHz).
#   2. Creates a Python virtualenv and installs core + Pi hardware deps.
#   3. Installs and enables the systemd service.
#
# Run from the repo root on the Pi:
#   chmod +x deploy/install_pi.sh
#   sudo ./deploy/install_pi.sh
#
# Re-running is safe (idempotent-ish).

set -euo pipefail

# --- resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_USER="${SUDO_USER:-pi}"
VENV_DIR="${REPO_DIR}/.venv"

echo "==> thermal-sentry Pi installer"
echo "    repo:    ${REPO_DIR}"
echo "    user:    ${SERVICE_USER}"
echo "    venv:    ${VENV_DIR}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run with sudo: sudo ./deploy/install_pi.sh" >&2
  exit 1
fi

# --- 1. enable I2C -----------------------------------------------------------
echo "==> Enabling I2C interface"
if command -v raspi-config >/dev/null 2>&1; then
  raspi-config nonint do_i2c 0 || true
fi

CONFIG_TXT="/boot/firmware/config.txt"
[[ -f "${CONFIG_TXT}" ]] || CONFIG_TXT="/boot/config.txt"
if [[ -f "${CONFIG_TXT}" ]]; then
  if ! grep -q "^dtparam=i2c_arm=on" "${CONFIG_TXT}"; then
    echo "dtparam=i2c_arm=on" >> "${CONFIG_TXT}"
  fi
  # 800 kHz bus is required for stable 16/32 Hz MLX90640 refresh rates.
  if ! grep -q "i2c_arm_baudrate=800000" "${CONFIG_TXT}"; then
    echo "dtparam=i2c_arm_baudrate=800000" >> "${CONFIG_TXT}"
  fi
  echo "    updated ${CONFIG_TXT} (reboot required to apply baudrate)"
fi

# Make sure the i2c-dev module is present.
modprobe i2c-dev || true

# --- 2. system packages + virtualenv ----------------------------------------
echo "==> Installing system packages"
apt-get update
apt-get install -y --no-install-recommends python3-venv python3-pip i2c-tools

echo "==> Creating virtualenv and installing Python dependencies"
sudo -u "${SERVICE_USER}" python3 -m venv "${VENV_DIR}"
sudo -u "${SERVICE_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip
sudo -u "${SERVICE_USER}" "${VENV_DIR}/bin/pip" install \
  -r "${REPO_DIR}/requirements.txt" \
  -r "${REPO_DIR}/requirements-pi.txt"
sudo -u "${SERVICE_USER}" "${VENV_DIR}/bin/pip" install --no-deps "${REPO_DIR}"

# --- 3. .env -----------------------------------------------------------------
if [[ ! -f "${REPO_DIR}/.env" ]]; then
  echo "==> Creating .env from .env.example (edit it to add webhook/email)"
  sudo -u "${SERVICE_USER}" cp "${REPO_DIR}/.env.example" "${REPO_DIR}/.env"
fi

# --- 4. systemd service ------------------------------------------------------
echo "==> Installing systemd service"
install -m 644 "${SCRIPT_DIR}/thermal-sentry.service" /etc/systemd/system/thermal-sentry.service
systemctl daemon-reload
systemctl enable thermal-sentry.service

echo ""
echo "==> Done."
echo "    Verify the sensor:   i2cdetect -y 1   (expect 0x33)"
echo "    Start the service:   sudo systemctl start thermal-sentry"
echo "    Tail logs:           journalctl -u thermal-sentry -f"
echo "    Dashboard:           http://<pi-ip>:8000"
echo ""
echo "    NOTE: if you changed the I2C baudrate, reboot before first run."
