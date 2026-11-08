from gtbench.config import load_config


def small_cfg():
    cfg = load_config()
    cfg["dataset"]["slices"] = ["normal", "low_texture"]
    cfg["dataset"]["frames_per_slice"] = 1
    cfg["dataset"]["validation_frames"] = 2
    cfg["stereo"]["max_disparity"] = 48
    return cfg
