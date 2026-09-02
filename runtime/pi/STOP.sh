#!/usr/bin/env bash
# STOP.sh - instantly stop the duck idle animation service and leave the robot
# PASSIVE (all servos limp, eyes off).  Worth having when the duck is animating
# on a desk and you want silence.
#
# Two layers:
#   1) stop the systemd service if it is installed/running  (needs sudo - it is
#      a system service).  This triggers the runner's graceful SIGTERM shutdown:
#      head eased to rest, torque off, antennas neutral, eyes off.
#   2) belt & braces: run idle_safe_off.py to force-disable torque on every
#      servo via raw pyserial (works WITHOUT sudo - the serial port is group
#      dialout, which clancey is in).  This guarantees passive even if the
#      service was not installed or was run by hand.
set -u

DUCK=~/duck
PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA095764-if00"

echo "[STOP] stopping duck-idle.service (if installed)..."
if systemctl list-unit-files 2>/dev/null | grep -q '^duck-idle\.service'; then
    if sudo -n systemctl stop duck-idle.service 2>/dev/null; then
        echo "[STOP]   service stopped (graceful shutdown)."
    else
        echo "[STOP]   need root to stop the service; run:  sudo systemctl stop duck-idle.service"
    fi
else
    echo "[STOP]   service not installed; skipping."
fi

# Kill any hand-run copy of the runner (e.g. a manual test), by exact PID.
for pid in $(pgrep -f 'idle_service\.py' 2>/dev/null); do
    echo "[STOP] sending SIGTERM to hand-run idle_service.py pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
done
sleep 2

echo "[STOP] belt & braces: forcing all servos torque-OFF (no sudo needed)..."
~/duck/venv/bin/python "$DUCK/idle_safe_off.py" "$PORT" || true

echo "[STOP] verifying passivity..."
if [ -f "$DUCK/bringup/final_verify.py" ]; then
    ~/duck/venv/bin/python "$DUCK/bringup/final_verify.py" 2>/dev/null | tail -3 || true
fi
echo "[STOP] done - robot should be passive (all torque OFF, eyes off)."
