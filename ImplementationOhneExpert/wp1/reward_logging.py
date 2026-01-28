from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardComponentsTensorboardCallback(BaseCallback):
    """
    Logs env info dict values (reward components) into TensorBoard.
    Works with VecEnv (info is a list).
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # For VecEnv, infos is list; take mean over envs for stability
        keys = [
            "r_safety", "r_energy", "r_comfort", "r_punctuality", "r_parking",
            "speed", "speed_limit", "delta_time_left", "max_jerk", "cummulated_energy"
        ]

        for k in keys:
            vals = [info.get(k) for info in infos if isinstance(info, dict) and (k in info)]
            if vals:
                self.logger.record(f"custom/{k}", float(np.mean(vals)))

        # Termination reasons: log counts when episodes end
        dones = self.locals.get("dones", None)
        if dones is not None:
            for d, info in zip(dones, infos):
                if d and isinstance(info, dict):
                    reason = str(info.get("termination_reason", "None"))
                    self.logger.record(f"custom/termination_{reason}", 1)

        return True
