"""MotionBricks interface to Kimodo's official SOMA30 hierarchy and assets."""

from kimodo.skeleton.definitions import SOMASkeleton30 as KimodoSOMA30


class SOMASkeleton30(KimodoSOMA30):
    def __init__(self, **kwargs):
        kwargs.pop("t_pose", None)
        super().__init__(**kwargs)
        self.t_pose = "standard"
        # Kimodo's reference assets are float64; the networks consume float32.
        self.float()
