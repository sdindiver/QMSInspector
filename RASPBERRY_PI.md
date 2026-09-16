# Running QMSInspector on a Raspberry Pi — Beginner's Guide

This guide gets the offline defect inspector running on a Raspberry Pi and viewable
in a web browser. It assumes **no prior Raspberry Pi or Linux experience**. Follow the
steps in order; each one is small.

> **What you'll end up with:** the Pi runs the inspector as a little website. You open
> that website in a browser (on the Pi, or on your laptop/phone on the same Wi-Fi),
> upload photos or a `.zip` of photos, and see **green = OK / red = DEFECT** with the
> defect circled on each image.

---

## 0. What you need (shopping list)

- **Raspberry Pi 4 (4 GB) or Raspberry Pi 5** — a Pi 5 is noticeably faster. *(A Pi with
  less than 4 GB RAM, or any 32-bit setup, will not work — see the note in Step 2.)*
- **microSD card, 32 GB or larger** (this is the Pi's "hard drive").
- **Official USB-C power supply** for the Pi.
- A way to write the SD card: your normal laptop/PC + an SD card reader.
- **Either** a monitor + keyboard + mouse for the Pi, **or** you access it "headless"
  from your laptop (both paths are covered below).

---

## 1. Put the operating system on the SD card

1. On your laptop, download and install **Raspberry Pi Imager** from
   <https://www.raspberrypi.com/software/>.
2. Insert the microSD card into your laptop.
3. Open Raspberry Pi Imager and choose:
   - **Device:** your Pi model (Pi 4 or Pi 5).
   - **Operating System:** *Raspberry Pi OS (64-bit)*. **The 64-bit version is
     required** — the AI libraries do not exist for the 32-bit version.
   - **Storage:** your SD card.
4. Click the **gear / "Edit Settings"** button before writing and set:
   - a **hostname** (e.g. `qmspi`),
   - a **username and password** (remember these!),
   - your **Wi-Fi name and password**,
   - **enable SSH** (lets you control the Pi from your laptop — handy if you go headless).
5. Click **Write** and wait. When done, put the SD card into the Pi and power it on.

---

## 2. Get to a command line on the Pi

You type commands into a black window called a **terminal**. Two ways to reach it:

- **With a monitor:** plug monitor + keyboard into the Pi. When it boots, open the
  **Terminal** app (black icon in the top bar).
- **Headless (from your laptop):** open a terminal on your laptop and run
  `ssh <username>@qmspi.local` (use the username and hostname you set in Step 1), then
  enter the password. You are now typing commands *on the Pi*.

> **Why 64-bit / 4 GB?** The inspector uses PyTorch and YOLO. Those only ship for
> 64-bit ARM, and they need roughly 1 GB of RAM to load the models. A 32-bit OS or a
> 1–2 GB Pi will fail to install or run out of memory.

---

## 3. Install the basic system tools (one time)

Copy-paste these lines into the Pi's terminal, one block at a time, pressing Enter:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git libopenblas0 libgl1
```

- `python3-pip`, `python3-venv` — Python and its package installer.
- `git` — used to download the project.
- `libopenblas0`, `libgl1` — small system libraries PyTorch/OpenCV need.

---

## 4. Download the project onto the Pi

```bash
git clone https://github.com/sankalpa-tech/QMSInspector.git
cd QMSInspector
```

You are now inside the project folder. (If you don't use GitHub, you can instead copy
the folder onto the Pi with a USB stick and `cd` into it.)

---

## 5. Create a private Python environment and install the app

A "virtual environment" keeps this project's libraries separate from the rest of the
system. Create and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt now starts with `(venv)`. Now install the inspector's libraries:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

- The first `pip install` gets the **CPU build of PyTorch** (the Pi has no NVIDIA GPU,
  so we use the CPU version — this line makes sure you get the right one).
- The second installs everything else the project lists (OpenCV, YOLO, Pillow, Flask…).

This step downloads a lot (~1–1.5 GB) and can take **10–30 minutes on a Pi** — that's
normal. Let it finish.

> **Tip (headless Pi with no monitor):** replace `opencv-python` behaviour by installing
> the lighter `opencv-python-headless` instead — it skips desktop/GUI libraries you don't
> need on a server. If you hit an OpenCV GUI error, run:
> `pip uninstall -y opencv-python && pip install opencv-python-headless`

---

## 6. Start the inspector

The project already includes a start script that cleans old runs and launches the
server:

```bash
./start.sh
```

(If that says "permission denied", run `chmod +x start.sh` once, then `./start.sh`
again. You can also just run `python qms.py serve --port 8000` directly.)

When it's running you'll see log lines and it will stay open. Leave this terminal window
running — that *is* the server. (To stop it, press `Ctrl + C`.)

---

## 7. Open the dashboard in a browser

- **On the Pi (with a monitor):** open Chromium and go to
  `http://localhost:8000`
- **From your laptop/phone (same Wi-Fi):** first find the Pi's address by running
  `hostname -I` on the Pi (it prints something like `192.168.1.42`). Then on your device
  open:
  `http://192.168.1.42:8000` (use *your* number).

You'll see the inspection page. **Upload one or more images, or a `.zip` of images**, and
you'll get, for each part:

- a **green OK** or **red DEFECT** verdict (defects shown first),
- the photo with the defect **circled and labeled** (e.g. "Serration Missing"),
- a summary count.

Every run is also **saved on the Pi** under `inspection_jobs/QMS-<date-time>/`
(`in/` = originals, `out/` = annotated images + JSON) so you keep a record.

---

## 8. (Optional) Make it start automatically on power-up

So the Pi becomes an appliance — power on, and the inspector is just *there* — create a
service:

```bash
sudo nano /etc/systemd/system/qms.service
```

Paste this (change `pi` to your username if different):

```ini
[Unit]
Description=QMS Inspector
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/QMSInspector
ExecStart=/home/pi/QMSInspector/venv/bin/python qms.py serve --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+O`, Enter, then `Ctrl+X`. Turn it on:

```bash
sudo systemctl enable qms.service
sudo systemctl start qms.service
```

Now the inspector runs on every boot with **no terminal needed**. Check it with
`sudo systemctl status qms.service`.

---

## How it works (the big picture)

```
 You (or a camera) provide a photo of a bracket
                    │
                    ▼
     Raspberry Pi  →  QMSInspector program:
                       1. finds the big mounting hole (YOLO)
                       2. crops it, checks the rim for serration teeth (classifier)
                       3. decides OK or DEFECT, draws the mark
                    │
                    ▼
     Web dashboard  →  green/red verdict + annotated photo
     (viewed in any browser on the network; also saved to disk)
```

- The program is **fully offline** — no internet or cloud needed after install.
- It processes images **one at a time**; a 22-image batch takes roughly **10–30 seconds
  on a Pi** (a Pi 5 is faster than a Pi 4). The first image after startup is slower
  because the AI models load once into memory.

---

## Will it run efficiently on a Pi?

Short answer: **yes — it will work, and it's efficient *for what it's designed to do*
(inspecting parts one at a time).** "Efficient" depends on your expectation, so here are
the real details.

### Why it's a good fit for a Pi

Your project is **lightweight by ML standards**:

- **Tiny models:** YOLOv8n (5.9 MB) + MobileNetV2 (8.7 MB). Both are the *small/mobile*
  variants — MobileNet was literally designed for phones and edge devices.
- **CPU-only, no GPU needed:** everything loads on the CPU already.
- **Modest RAM:** ~1 GB resident. A 4 GB Pi has plenty of headroom.
- **Recent optimization:** removing the `hole_rough` / `HoughCircles` step cut the heaviest
  CPU operation — that speedup helps the slower Pi CPU proportionally *more* than a fast
  desktop.

### Realistic speed

On a fast x86 dev machine it's ~0.43s/image (warm). A Pi's CPU is several times slower, so
expect roughly:

| Board | Per image (warm) | First image (cold model load) |
|-------|------------------|-------------------------------|
| **Raspberry Pi 5** | ~1.5–3s | ~15–25s |
| **Raspberry Pi 4** | ~3–8s  | ~25–40s |

So a **22-image batch ≈ 1–3 minutes on a Pi.** The cold delay happens only **once per
startup** (models load into memory), not per image.

### What "efficient" means for this use case

- ✅ **Efficient** if inspection is: operator places a part → snap → wait a couple seconds
  → red/green. That's exactly this project's workflow. A few seconds per part is completely
  fine on a line.
- ❌ **Not** efficient if you expect **real-time video** (30 frames/sec) or **hundreds of
  parts per minute** — a Pi CPU can't do that.

### If you ever need it faster

1. **Use a Pi 5** over a Pi 4 (biggest easy win).
2. **Add an AI accelerator** — Google Coral or Hailo-8 (a small add-on module) can run
   YOLO/MobileNet 10–50× faster than the Pi CPU.
3. **Keep the server running** (the autostart service in Step 8) so you pay the cold-load
   cost once, not on every batch.

### Bottom line

For an **offline, point-of-inspection appliance checking parts one-by-one**, this project
runs **efficiently enough on a Raspberry Pi 4, and comfortably on a Pi 5.** It's small,
low-power, and the models are already the edge-optimized kind. The only thing a Pi *can't*
give you is high-throughput / real-time speed — and this use case doesn't need that.

---

## Frequently asked beginner questions

**Do I need internet on the Pi?** Only during install (Steps 3–5, to download things).
After that it runs completely offline.

**How does the camera fit in?** Today the app works by **uploading** photos through the
web page. Making a camera snap-and-inspect *automatically* is a small extra script on top
(a Pi Camera or USB camera that saves a photo, which the inspector then reads). Ask if you
want that set up — the detection itself is already done.

**Can several people see it?** Yes — anyone on the same Wi-Fi/LAN can open
`http://<pi-address>:8000`. Note there is **no password/login**, so only use it on a
trusted local network, never exposed to the public internet.

**It's slow / runs out of memory.** Make sure you used **64-bit** Raspberry Pi OS and a
**4 GB (or Pi 5)** board. Close other apps. A Pi 5 or an AI accelerator (Coral/Hailo) helps
if you need more speed.

**Nothing loads in the browser.** Check the `./start.sh` terminal is still running, that
you used the Pi's real address from `hostname -I`, and that your device is on the same
Wi-Fi.
