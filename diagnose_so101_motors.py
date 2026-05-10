"""Scan and ping the SO-101 Feetech motors on COM5."""

import os

if os.name == "nt":
    os.environ.pop("SSLKEYLOGFILE", None)

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


PORT = "COM5"
ROBOT_ID = "my_awesome_follower_arm"


def main() -> None:
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    bus = robot.bus

    print(f"Opening motor bus on {PORT}...")
    bus.connect(handshake=False)
    try:
        print("\nExpected motors:")
        for name, motor in bus.motors.items():
            print(f"- {name}: id={motor.id}, model={motor.model}")

        print("\nPing expected motor IDs:")
        for name, motor in bus.motors.items():
            model = bus.ping(name, num_retry=3, raise_on_error=False)
            status = "OK" if model is not None else "NO RESPONSE"
            print(f"- {name} id={motor.id}: {status} ({model})")

        print("\nBroadcast scan by baudrate:")
        bus.port_handler.closePort()
        scan = type(bus).scan_port(PORT)
        if not scan:
            print("No motors found.")
        for baudrate, ids in scan.items():
            print(f"- {baudrate}: ids={ids}")
    finally:
        if bus.is_connected:
            bus.port_handler.closePort()


if __name__ == "__main__":
    main()
