#!/usr/bin/env python3
"""Force every Feetech servo torque-OFF (reg 40 = 0) via raw pyserial.

Dependency-light passivity backstop for the duck-idle service: it does NOT need
rustypot or open_duck_anim, only pyserial (already in the venv).  Safe to run at
any time - it only writes Torque_Enable=0, it never commands a goal position, so
nothing moves; servos simply go limp.  Used as the systemd ExecStopPost belt &
braces and by STOP.sh.

Exit 0 always (best effort) unless the port cannot be opened at all.
"""
import sys
import time

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA095764-if00"
BAUD = 1_000_000
ALL_IDS = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33]


def _cksum(vals):
    return (~sum(vals)) & 0xFF


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    try:
        import serial
    except Exception as e:  # noqa: BLE001
        print("idle_safe_off: pyserial unavailable:", e)
        return 0
    try:
        ser = serial.Serial(port, BAUD, timeout=0.1)
    except Exception as e:  # noqa: BLE001
        print("idle_safe_off: cannot open %s: %s (nothing to do)" % (port, e))
        return 0
    try:
        time.sleep(0.05)
        for sid in ALL_IDS:
            pkt = [0xFF, 0xFF, sid, 0x04, 0x03, 40, 0x00]
            pkt.append(_cksum(pkt[2:]))
            try:
                ser.write(bytes(pkt))
                time.sleep(0.003)
                ser.reset_input_buffer()
            except Exception:  # noqa: BLE001
                pass
        print("idle_safe_off: torque-disable sent to all 14 servos on %s" % port)
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
