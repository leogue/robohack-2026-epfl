"""Measure plant humidity with an ESP, then run the watering VLA on the dry side."""

import os

if os.name == "nt":
    os.environ.pop("SSLKEYLOGFILE", None)

import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from esp_shared import read_latest_sample
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.scripts import lerobot_record as lr
from lerobot.scripts.lerobot_record import DatasetRecordConfig, PreTrainedConfig, RecordConfig

from replay import load_motion_dataset, make_replay_robot, replay_loaded_motion


ESP_WARMUP_S = 0.5
ESP_MAX_AGE_S = 3.0

ROBOT_PORT = "COM5"
ROBOT_ID = "my_awesome_follower_arm"

POLICY_PATH = "lleeoogg/robothack_give_water_smolvla_10mai_6h15"
#POLICY_PATH ="lleeoogg/robothack_give_water_smolvla"
POLICY_DEVICE = "cuda"
EPISODE_TIME_S = 60
DISPLAY_DATA = True
STREAMING_ENCODING = True
ENCODER_THREADS = 2
VCODEC = "auto"

# ESP logic: if moisture stays at 0 while probing, the plant is dry.
WET_THRESHOLD = 0

RENAME_MAP = {
    "observation.images.top_cam": "observation.images.camera1",
    "observation.images.wrist_cam": "observation.images.camera2",
}

TASKS = {
    "left": "Pick up the water bottle and pour water into the left flower pot.",
    "right": "Pick up the water bottle and pour water into the right flower pot.",
}

EVAL_REPO_IDS = {
    "left": "lleeoogg/eval_give_water_left",
    "right": "lleeoogg/eval_give_water_right",
}

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
CACHED_POLICY = None
ORIGINAL_MAKE_POLICY = lr.make_policy


@dataclass
class Sample:
    timestamp: float
    light_pct: int
    temp_c: float
    moisture_pct: int


@dataclass
class Measurement:
    side: str
    samples: list[Sample] = field(default_factory=list)

    @property
    def moisture_values(self) -> list[int]:
        return [sample.moisture_pct for sample in self.samples]

    @property
    def max_moisture(self) -> int:
        return max(self.moisture_values, default=0)

    @property
    def avg_moisture(self) -> float:
        values = self.moisture_values
        return statistics.fmean(values) if values else 0.0

    @property
    def is_wet(self) -> bool:
        return self.max_moisture > WET_THRESHOLD


class ESPStateReader:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active_side: str | None = None
        self._samples_by_side: dict[str, list[Sample]] = {"left": [], "right": []}
        self._thread: threading.Thread | None = None
        self._last_timestamp: float | None = None

    def __enter__(self) -> "ESPStateReader":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def start_measurement(self, side: str) -> None:
        with self._lock:
            self._samples_by_side[side] = []
            self._active_side = side

    def stop_measurement(self, side: str) -> Measurement:
        with self._lock:
            if self._active_side == side:
                self._active_side = None
            return Measurement(side=side, samples=list(self._samples_by_side[side]))

    def _run(self) -> None:
        while not self._stop.is_set():
            sample_dict = read_latest_sample(max_age_s=ESP_MAX_AGE_S)
            if sample_dict is None:
                time.sleep(0.1)
                continue

            timestamp = float(sample_dict["timestamp"])
            if self._last_timestamp == timestamp:
                time.sleep(0.05)
                continue
            self._last_timestamp = timestamp

            sample = Sample(
                timestamp=timestamp,
                light_pct=int(sample_dict["light"]),
                temp_c=float(sample_dict["temp"]),
                moisture_pct=int(sample_dict["moisture"]),
            )
            print(
                f"ESP light={sample.light_pct}% temp={sample.temp_c:.1f}C "
                f"moisture={sample.moisture_pct}%"
            )

            with self._lock:
                if self._active_side is not None:
                    self._samples_by_side[self._active_side].append(sample)


def load_policy_config() -> PreTrainedConfig:
    print(f"Loading policy config from {POLICY_PATH}...")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.device = POLICY_DEVICE
    policy_cfg.pretrained_path = POLICY_PATH
    return policy_cfg


def make_cameras() -> dict[str, OpenCVCameraConfig]:
    return {
        "top_cam": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        "wrist_cam": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }


def make_robot_config() -> SO101FollowerConfig:
    return SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=make_cameras(),
        disable_torque_on_disconnect=False,
        use_degrees=True,
    )


def preload_policy(policy_cfg: PreTrainedConfig) -> None:
    global CACHED_POLICY

    print("Preloading VLA weights...")
    robot = lr.make_robot_from_config(make_robot_config())
    teleop_action_processor, _, robot_observation_processor = lr.make_default_processors()
    dataset_features = lr.combine_feature_dicts(
        lr.aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=lr.create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        lr.aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=lr.create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    warmup_root = Path("outputs") / "preload_policy_metadata" / RUN_ID
    dataset = lr.LeRobotDataset.create(
        f"lleeoogg/preload_policy_{RUN_ID}",
        fps=30,
        root=warmup_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        streaming_encoding=False,
    )
    try:
        CACHED_POLICY = ORIGINAL_MAKE_POLICY(policy_cfg, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    finally:
        dataset.finalize()

    def make_policy_cached(cfg, ds_meta=None, env_cfg=None, rename_map=None):
        if cfg.pretrained_path == POLICY_PATH and CACHED_POLICY is not None:
            print("Using preloaded VLA policy.")
            return CACHED_POLICY
        return ORIGINAL_MAKE_POLICY(cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)

    lr.make_policy = make_policy_cached
    print("VLA preloaded.")


def preload_motion_datasets():
    return {
        "left": load_motion_dataset("left"),
        "right": load_motion_dataset("right"),
    }


def measure_side(reader: ESPStateReader, side: str, dataset, robot) -> Measurement:
    print(f"\nMeasuring {side} plant...")
    reader.start_measurement(side)
    try:
        replay_loaded_motion(side, dataset, robot)
    finally:
        measurement = reader.stop_measurement(side)

    print_measurement(measurement)
    return measurement


def print_measurement(measurement: Measurement) -> None:
    state = "wet" if measurement.is_wet else "dry"
    print(
        f"{measurement.side}: {state} "
        f"(samples={len(measurement.samples)}, max={measurement.max_moisture}%, "
        f"avg={measurement.avg_moisture:.1f}%)"
    )


def choose_plant_to_water(left: Measurement, right: Measurement) -> str | None:
    if left.is_wet and right.is_wet:
        return None
    if not left.is_wet and right.is_wet:
        return "left"
    if left.is_wet and not right.is_wet:
        return "right"

    return "left" if left.avg_moisture <= right.avg_moisture else "right"


def make_record_config(side: str, policy_cfg: PreTrainedConfig) -> RecordConfig:
    dataset_cfg = DatasetRecordConfig(
        repo_id=f"{EVAL_REPO_IDS[side]}_{RUN_ID}",
        single_task=TASKS[side],
        num_episodes=1,
        episode_time_s=EPISODE_TIME_S,
        push_to_hub=False,
        streaming_encoding=STREAMING_ENCODING,
        encoder_threads=ENCODER_THREADS,
        vcodec=VCODEC,
        rename_map=RENAME_MAP,
    )
    return RecordConfig(
        robot=make_robot_config(),
        dataset=dataset_cfg,
        policy=policy_cfg,
        display_data=DISPLAY_DATA,
        play_sounds=True,
    )


def water_plant(side: str, policy_cfg: PreTrainedConfig) -> None:
    print(f"\nSelected plant: {side}")
    print(f"Task: {TASKS[side]}")
    lr.record(make_record_config(side, policy_cfg))


def main() -> None:
    policy_cfg = load_policy_config()
    preload_policy(policy_cfg)
    motion_datasets = preload_motion_datasets()
    input("\nVLA is ready. Press ENTER to start the humidity pipeline...")

    if read_latest_sample(max_age_s=ESP_MAX_AGE_S) is None:
        raise RuntimeError("No fresh ESP data. Start `python3 .\\esp_serial_broker.py` first.")

    with ESPStateReader() as reader:
        time.sleep(ESP_WARMUP_S)
        robot = make_replay_robot()
        try:
            left = measure_side(reader, "left", motion_datasets["left"], robot)
            right = measure_side(reader, "right", motion_datasets["right"], robot)
        finally:
            robot.disconnect()

    side = choose_plant_to_water(left, right)
    if side is None:
        print("\nBoth plants look wet. Skipping watering.")
        return

    water_plant(side, policy_cfg)


if __name__ == "__main__":
    main()
