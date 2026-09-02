# duck-idle — install the boot-time idle "alive" animation service

Power on the duck → it comes alive on its own: head moving gently, eyes blinking,
**legs limp**, no laptop / SSH / manual command. Runs as a supervised systemd
**system** service (starts without a login session).

Everything is already prepared and hardware-tested under user `clancey`. The only
thing left needs **root** (installing to `/etc` and enabling at boot). Run the one
block below. It is **idempotent** — safe to re-run after editing the unit or the
runner.

- Runner:      `/home/clancey/duck/idle_service.py`   (head + eyes only; legs never torqued)
- Backstop:    `/home/clancey/duck/idle_safe_off.py`  (raw torque-off, ExecStopPost)
- Unit source: `/home/clancey/duck/duck-idle.service`
- Stop helper: `/home/clancey/duck/STOP.sh`

---

## 1. Install + enable + start at boot  (run as root)

Copy-paste this whole block into an SSH session on `BdxBot.local`:

```bash
sudo install -m 0644 -o root -g root \
    /home/clancey/duck/duck-idle.service \
    /etc/systemd/system/duck-idle.service
sudo systemctl daemon-reload
sudo systemctl enable --now duck-idle.service
sleep 3
systemctl --no-pager --full status duck-idle.service | head -n 20
```

You should see `Active: active (running)` and, within a couple of seconds, the
head start to move and the eyes blink. Legs stay limp.

Watch it live:

```bash
journalctl -u duck-idle.service -f
```

(Ctrl-C just stops following the log; it does **not** stop the service.)

---

## 2. Stop it now (leave robot passive)

Any of these. All leave the robot **passive** (all torque OFF, eyes off):

```bash
sudo systemctl stop duck-idle.service     # graceful: head eased to rest, torque off
# or, belt & braces (also works with no sudo, also kills a hand-run copy):
/home/clancey/duck/STOP.sh
```

`sudo systemctl stop` sends SIGTERM; the runner eases the head to rest, disables
torque, neutralises the antennas and turns the eyes off, then `ExecStopPost`
force-disables every servo as a backstop.

---

## 3. Disable autostart entirely (stop it starting on boot)

```bash
sudo systemctl disable --now duck-idle.service
```

`--now` also stops the running instance. To re-enable later:
`sudo systemctl enable --now duck-idle.service`.

To remove completely:

```bash
sudo systemctl disable --now duck-idle.service
sudo rm -f /etc/systemd/system/duck-idle.service
sudo systemctl daemon-reload
```

---

## 4. Tunables (optional)

Edit `/etc/systemd/system/duck-idle.service` (uncomment the `Environment=` lines),
then `sudo systemctl daemon-reload && sudo systemctl restart duck-idle.service`:

- `DUCK_IDLE_ACTIVE_S=480`  animate this long before a relax/thermal checkpoint
- `DUCK_IDLE_RELAX_S=20`    head fully de-energised (limp) + thermal scan each checkpoint
- `DUCK_IDLE_ANTENNAS=0`    set `1` to also play the antenna show (owner: noisy → default off)

---

## 5. Safety notes (why this is safe to leave unattended)

- **Head + eyes only.** The legs are *never* torqued by this service. If the duck
  is picked up, knocked, or sitting oddly at boot, nothing fights it.
- **Clean stop.** SIGTERM / SIGHUP / atexit all ease the head to rest and torque
  off; `ExecStopPost` is a raw backstop. `systemctl stop` will not leave the head
  energised.
- **×0.5 derated envelope**, enforced by the animation engine (not bypassed).
- **Thermal duty-cycle guard.** Every ~8 min the head goes fully limp for ~20 s
  and a thermal/voltage/error scan is logged; abort thresholds trip a clean
  passive exit.
- **Fail safe, not loud.** Missing serial bus / failed clip / any startup error →
  clean exit leaving hardware passive. `Restart=on-failure`, `RestartSec=10`,
  `StartLimitBurst=4` in 300 s → a persistent fault **stops** rather than looping
  and thrashing servos.
- **Waits for hardware.** `Wants=/After=` the CH343 by-id `.device` unit, plus the
  runner polls for the serial device before touching anything.
