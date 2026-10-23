# Sleepwalk detector

This project watches a Kinect for Windows v2 sensor (Xbox One Kinect, USB id `VID_045E` / `PID_02D8`) and saves a clip when someone walks through the view after you have gone to bed.

## Prerequisites

- Windows 10 or 11. The Kinect v2 SDK does not run on macOS or Linux.
- Python 3.12 or newer, with `python.exe` on PATH. The installer is at [python.org/downloads](https://www.python.org/downloads/). Check "Add python.exe to PATH".
- Kinect for Windows SDK 2.0, installed by an administrator. That install provides the USB driver for the whole machine. The program cannot install the driver itself. Uninstall steps are in [UNINSTALL.md](UNINSTALL.md).
- An Xbox One Kinect (Kinect v2), its power adapter, and a USB 3.0 port. This laptop's built-in webcam is a different camera and is not used.
- Only one program can use the Kinect at a time. Close the viewer before starting the detector.
- An internet connection the first time you start the program. It creates `.venv` and downloads the Python packages, including PyTorch. That download is large.
- An NVIDIA GPU is optional. Training and clip checking use it when it is present. If it is missing, clip checking uses an Intel NPU when one is present, and training stays on the CPU. With neither, both use the CPU.

## Sleepwalk detector

From this folder, double-click `run.bat`, or run:

```powershell
python .\sleepwalk_detector.py
```

Windows asks for administrator approval each time the program starts. Choose Yes. The first launch installs the packages into `.venv` and then opens the window. Later launches skip the install. Detect and Train are the same window. Use the switch at the top. This command opens Train instead:

```powershell
python .\train_person_model.py
```

The live picture has **Infrared + joints** and **Color + joints**. Color is easier to recognize in a lit room. Recorded clips and the person check always use infrared, because that still works in the dark. The Kinect lights the scene itself.

1. Aim the sensor at the doorway, hallway, or side of the bed you want to watch.
2. Check the live picture. Joints appear when a person is standing in view, about 0.5 to 4.5 meters away.
3. Press **Start detector**.
4. You get a grace period, 2 minutes unless you change it in **Settings**. The countdown is on the window. Walk out of view and get into bed during this time. Movement during the grace period is not saved.
5. When the countdown ends, the status changes to **Monitoring**.
6. If a person is in view and **walking**, recording starts. Standing still does not start a clip. A skeleton that is only a short cloud of points, a body lying flat, or a set of impossible limb lengths does not count as walking. Walking means the tracked body moves at least 0.4 meters over about a second, unless you change that distance in **Settings**.
7. The saved clip includes the buffer set in **Settings** (10 seconds before the walk and 10 seconds after the person leaves, unless you change them). If they pause but stay visible, the same clip keeps going.
8. Press **Stop** to end monitoring. Each clip is scored, then a report window opens. You can play each clip and export the whole report as a PDF or a CSV. A clip is **Person**, **No person**, or **Unfeasible**. **Clear History** removes those clips and their scores. It leaves the training pictures and the person model in place. Close the window to quit.

**Settings** is at the top right of the window, and Control-comma opens it. It stores the grace period, the clip buffers, the walk distance, the trainer shutter countdown, and the body-size limits used for the Unfeasible label. The window follows the Windows light or dark appearance.

Clips are written to the `clips` folder next to this file, named like `sleepwalk_20261022_003015.mp4`. **Open clips folder** opens that folder. Each clip is the infrared picture with the skeleton drawn on it. Raw infrared frames, without the skeleton, are saved beside the clip in a folder ending in `_frames`. Those raw frames are what the person model looks at.

## Train the person model

The model answers a narrower question than the skeleton: is a person actually visible in the infrared picture? A glitch can still draw a bundle of joints. The model ignores those points and looks at the camera image.

Collect and label pictures in the trainer, before you train. You want both kinds, and they should not all look the same. Move a step, turn, sit, and stand farther away between person pictures. For empty pictures, leave the room and let it see the bed, doorway, and furniture. The detector does not save training pictures.

The label is the folder you choose. **Person** pictures teach “this is a standing person.” **Unfeasible** pictures teach “a person is here, but they are lying down or otherwise not standing.” **Empty room** pictures teach “this is not a person.” The model does not invent those labels.

1. Open the window and switch to **Train**.
2. Click **Person**, **Unfeasible**, or **Empty room**.
3. Click **Photo** or **Video**. There is a countdown so you can step into place. It is 5 seconds unless you change it in **Settings**. A photo saves one infrared picture. A video records until you press **Stop video**, and about one frame a second is also saved as a training picture. The live view shows the skeleton. The saved pictures do not.
4. Collect at least 15 of each, and 40 or more is better. For Person, include your back and your side while standing, not only facing the camera. For Unfeasible, lie down the way you would in bed. The window shows the counts.
5. Click **Train**. The window shows how many person and empty pictures it found. The first training run downloads a small starting model (MobileNet) and then adapts it to your infrared pictures. Progress and accuracy print in the window. It saves `models\person_classifier.pt`.

Training and clip checking run on this computer. Nothing is sent out to be scored. The program uses the NVIDIA GPU when it is available. If that GPU is missing, clip checking uses the NPU, and training stays on the CPU because the NPU cannot run the training step. If neither the GPU nor the NPU is available, both use the CPU.

6. Switch back to **Detect**. When you press **Stop**, every saved clip is scored and the report opens. The list shows **Person**, **No person**, or **Unfeasible**. Person means the infrared model saw a standing human. Unfeasible means it saw someone lying down, or the skeleton was a short cloud, a body lying flat, or limb lengths that cannot be a person.

Add more pictures and train again whenever the flags look wrong. Clips recorded before this raw-frame saving was added are scored from the video itself, which still has the skeleton drawn on it, so treat those older scores as rough.

## Manual viewer

The other window is for looking at the cameras yourself. It does not record.

```powershell
python .\kinect_view.py
```

Close the window to quit.

## Views

The window has a button for each view and checkboxes so you can combine them.

| Control | What it shows |
| --- | --- |
| Color | Color camera |
| Infrared | Infrared camera |
| Infrared + joints | Infrared camera with body joints drawn on it |
| Color + joints | Color camera with body joints drawn on it |
| All three | Color and infrared side by side, with joints on both |

Checking **Color** and **Infrared** together puts those pictures side by side. **Joints** draws the skeleton on every picture that is visible. Tracked joints are filled circles. Inferred joints are smaller rings. Each person gets a different color.

**Colorize infrared** tints the infrared picture. Leave it off for a grayscale infrared image.

The status line shows the resolution of each open stream and how many bodies are tracked. Joint tracking needs a person in front of the sensor, roughly 0.5 to 4.5 meters away, with the head and joints in view.
