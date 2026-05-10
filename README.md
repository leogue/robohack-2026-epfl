# ADAM - RoboHack 2026 EPFL

ADAM is a robotic gardening prototype built for the RoboHack 2026 EPFL hardware track.

The project combines an SO-101 robotic arm, an ESP32-S3 sensor node, a small web dashboard, and a SmolVLA policy fine-tuned for watering plants. The robot first probes the soil humidity of the left and right plants, decides which one is dry, then runs the vision-language-action model with the matching instruction.

## Hardware

- SO-101 follower arm on `COM5`
- ESP32-S3 sensor board on `COM7`
- Two OpenCV cameras:
  - top camera: index `1`
  - wrist camera: index `2`
- Sensors:
  - light percentage
  - temperature
  - soil humidity

## Main Workflow

Start the ESP serial broker first. It is the only process that opens `COM7`; the robot workflow and web dashboard read shared sensor values from `outputs/esp_latest.json`.

```powershell
python3 .\esp_serial_broker.py
```

Start the web dashboard:

```powershell
python3 .\web_lerobot_epfl\robohack-plant\dashboard.py
```

Open the URL printed by the server, usually:

```text
http://127.0.0.1:8001
```

Run the full robot workflow:

```powershell
$env:PYTHONIOENCODING="utf-8"
python3 .\water_from_humidity.py
```

The script preloads the VLA model, waits for `ENTER`, measures the left and right plants, then waters the dry side.

## Debug Scripts

Run the watering policy directly:

```powershell
python3 .\debug_smolvla_left.py
python3 .\debug_smolvla_right.py
```

Replay only the humidity probing motions:

```powershell
python3 .\replay.py left
python3 .\replay.py right
```

Check SO-101 motor IDs on `COM5`:

```powershell
python3 .\diagnose_so101_motors.py
```

## Notes

- Only `esp_serial_broker.py` should access `COM7`.
- The web dashboard is named ADAM.
- Dataset names are generated with timestamps to avoid local Hugging Face cache collisions.
- `disable_torque_on_disconnect=False` is used to avoid Feetech gripper overload errors during shutdown.
