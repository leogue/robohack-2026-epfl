"""Own COM7 and publish the latest ESP sensor sample for other processes."""

import os

if os.name == "nt":
    os.environ.pop("SSLKEYLOGFILE", None)

import serial

from esp_shared import parse_esp_line, write_latest_sample


PORT = "COM7"
BAUDRATE = 115200


def main() -> None:
    print(f"Opening ESP serial broker on {PORT} @ {BAUDRATE}. Press Ctrl+C to stop.")
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        while True:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                continue

            sample = parse_esp_line(raw)
            if sample is None:
                print(f"ESP ignored: {raw}")
                continue

            write_latest_sample(sample)
            print(
                f"ESP light={sample['light']}% temp={sample['temp']:.1f}C "
                f"moisture={sample['moisture']}%"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
