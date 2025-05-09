from pathlib import Path
import yaml


def load_config(config_file, r):
    config_path = Path(r.Rlocation(config_file))
    config_dir = config_path.parent
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        for name in config.get("defaults", []):
            path = r.Rlocation(f"{config_dir / name}")
            with open(path, "r") as pf:
                parent_config = yaml.safe_load(pf)
                config.update(parent_config)
    if "defaults" in config:
        del config["defaults"]
    return config
