"""Debug script: run the SmolVLA watering policy on the left plant."""

from water_from_humidity import load_policy_config, preload_policy, water_plant


def main() -> None:
    policy_cfg = load_policy_config()
    preload_policy(policy_cfg)
    input("\nVLA is ready. Press ENTER to water the left plant...")
    water_plant("left", policy_cfg)


if __name__ == "__main__":
    main()
