"""
Utilities for logging progress metrics and saving checkpoints.
"""

import logging
from pathlib import Path
from time import time, strftime, gmtime

from utils.storage import save


class Log:
    def __init__(self, name, print_freq=10, save_freq=1000):
        self.start = -1
        self.name = name
        self.print_freq = print_freq
        self.save_freq = save_freq
        Path("../trained_anmls").mkdir(exist_ok=True)

    def __call__(self, it, loss, acc, model):
        if self.start < 0:
            self.start = time()
        else:
            if it % self.print_freq == 0:
                end = time()
                elapsed = end - self.start
                self.start = end
                logging.info(f"{it}: {loss.item():.3f} | {acc:.3f} ({strftime('%H:%M:%S', gmtime(elapsed))})")

            if it % self.save_freq == 0:
                save(model, f"trained_anmls/{self.name}-{it}.net")

    def close(self, it, model):
        save(model, f"trained_anmls/{self.name}-{it}.net")
