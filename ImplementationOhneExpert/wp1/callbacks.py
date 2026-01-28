from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardBreakdownCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_metrics = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info:
                self.ep_metrics.append(info)

        dones = self.locals.get("dones", [])
        if len(dones) > 0 and dones[0]:
            # last info = terminal
            last = self.ep_metrics[-1] if self.ep_metrics else {}
            for k in ["r_safety","r_energy","r_comfort","r_punctuality","r_parking"]:
                if k in last:
                    self.logger.record(f"reward_components/{k}", float(last[k]))
            if "termination_reason" in last:
                self.logger.record("episode/termination_reason", 1)  # TB can't log strings cleanly
            self.ep_metrics = []
        return True

