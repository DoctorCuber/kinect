# Uninstall the Kinect driver and SDK

The camera driver is installed for Windows, not inside `.venv`. Removing it does not delete this folder, the virtual environment, or `kinect_view.py`.

## Camera driver only

The driver package is **Kinect for Windows Drivers v2.0_1409 (DAILY)**.

1. Open **Settings > Apps > Installed apps**.
2. Uninstall **Kinect for Windows Drivers v2.0_1409 (DAILY)**.

Or from an Administrator PowerShell window:

```powershell
Start-Process msiexec.exe -ArgumentList '/X{7C9F054E-F742-4DAD-B7E0-9A280F1F0ACB}','/quiet','/norestart' -Verb RunAs -Wait
```

After this, Device Manager no longer shows **WDF KinectSensor Interface 0**. The microphone can remain, because Windows uses a separate USB audio driver for it. The SDK runtime can remain installed, but the viewer will not be able to open the camera.

## Driver, runtime, and SDK together

Uninstall **Kinect for Windows SDK v2.0_1409**. That removes the camera driver, the Kinect runtime, and the SDK tools.

1. Open **Settings > Apps > Installed apps**.
2. Uninstall **Kinect for Windows SDK v2.0_1409**.

Or from an Administrator PowerShell window:

```powershell
& "C:\ProgramData\Package Cache\{2f7f3dc4-de9a-4605-821f-b686f26392d8}\KinectSDK-v2.0_1409-Setup.exe" /uninstall /quiet
```

That bundle also removes these packages if they are still listed:

- Kinect for Windows SDK v2.0
- Kinect for Windows Runtime v2.0_1409
- Kinect for Windows Drivers v2.0_1409 (DAILY)

## Leftover driver in the driver store

Do this only if Device Manager still lists **WDF KinectSensor Interface 0** or **Xbox NUI Sensor** after the uninstall above.

1. Open **Device Manager**.
2. Right-click the Kinect camera entry, choose **Uninstall device**, and check **Attempt to remove the driver for this device**.
3. Unplug the sensor.

To delete a driver package Windows kept, from an Administrator PowerShell window:

```powershell
pnputil /enum-drivers | Select-String -Pattern "kinect" -Context 8,2
```

Note the `oem##.inf` name in the Kinect block, then:

```powershell
pnputil /delete-driver oem##.inf /uninstall /force
```

Replace `oem##.inf` with the published name from that list.

## If the camera stops opening later

Memory Integrity (Core isolation) was turned on when the driver was installed, and the driver loaded with it on. If a later Windows update blocks the driver, open **Windows Security > Device security > Core isolation details**, turn **Memory integrity** off, and restart. After the Kinect driver is uninstalled, turn Memory Integrity back on and restart again.
