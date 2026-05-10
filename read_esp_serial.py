"""Read and print serial data from an ESP on COM7 at 115200 baud."""

import serial


PORT = "COM7"
BAUDRATE = 115200


def main() -> None:
    print(f"Opening {PORT} at {BAUDRATE} baud. Press Ctrl+C to stop.")

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
            while True:
                data = ser.readline()
                if data:
                    print(data.decode("utf-8", errors="replace").rstrip())
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
