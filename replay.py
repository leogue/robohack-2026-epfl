"""
Replay a recorded motion from Hugging Face Hub.

Usage:
    python replay.py left
    python replay.py right
"""

import os
import sys
import time

if os.name == "nt":
    os.environ.pop("SSLKEYLOGFILE", None)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ModuleNotFoundError:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


MOTIONS = {
    "left": "lleeoogg/motion_get_humidity_left",
    "right": "lleeoogg/motion_get_humidity_right",
}


def load_motion_dataset(motion_name: str) -> LeRobotDataset:
    if motion_name not in MOTIONS:
        raise ValueError(f"Unknown motion '{motion_name}'. Available: {list(MOTIONS.keys())}")

    repo_id = MOTIONS[motion_name]
    print(f"Loading dataset {repo_id}...")
    return LeRobotDataset(repo_id)


def make_replay_robot() -> SO101Follower:
    print("Connecting to robot...")
    config = SO101FollowerConfig(
        port="COM5",
        id="my_awesome_follower_arm",
        disable_torque_on_disconnect=False,
        use_degrees=True,
    )
    robot = SO101Follower(config)
    try:
        robot.connect()
    except ConnectionError as exc:
        if "Failed to write 'Lock'" not in str(exc):
            raise
        print("Motor configure failed while writing Lock; continuing without reconfiguration.")
        if not robot.bus.is_connected:
            robot.bus.connect()
        for motor in robot.bus.motors:
            try:
                robot.bus.write("Torque_Enable", motor, 1, num_retry=5)
            except ConnectionError as torque_exc:
                print(f"Warning: could not enable torque on {motor}: {torque_exc}")

    print(f"   Motors: {list(robot.bus.motors.keys())}")
    return robot


def replay_loaded_motion(
    motion_name: str,
    dataset: LeRobotDataset,
    robot: SO101Follower,
    episode_index: int = 0,
) -> None:
    motor_names = list(robot.bus.motors.keys())

    from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    to_idx = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    n_frames = to_idx - from_idx
    fps = dataset.meta.fps
    dt = 1.0 / fps

    print(f"Replaying '{motion_name}' ({n_frames} frames at {fps} Hz)...")
    start = time.time()

    for i in range(from_idx, to_idx):
        loop_start = time.time()
        frame = dataset[i]
        action_tensor = frame["action"]

        action_dict = {
            f"{name}.pos": float(action_tensor[j])
            for j, name in enumerate(motor_names)
        }
        robot.send_action(action_dict)

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    total = time.time() - start
    print(f"Done in {total:.2f}s")


def replay_motion(motion_name: str, episode_index: int = 0) -> None:
    dataset = load_motion_dataset(motion_name)
    robot = make_replay_robot()
    try:
        replay_loaded_motion(motion_name, dataset, robot, episode_index=episode_index)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay.py [left|right]")
        sys.exit(1)
    replay_motion(sys.argv[1])
