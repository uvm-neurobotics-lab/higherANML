import logging
from pathlib import Path
from time import time, strftime, gmtime
from torch import save


def unzip(l):
    # turn list of pairs into pairs of lists
    return list(zip(*l))


# Yield successive n-sized chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


class Log:
    def __init__(self, name, print_freq=10, save_freq=1000):
        self.start = -1
        self.name = name
        self.print_freq = print_freq
        self.save_freq = save_freq
        Path("./trained_anmls").mkdir(exist_ok=True)

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
                save(model.state_dict(), f"trained_anmls/{self.name}-{it}.pth")

    def close(self, it, model):
        save(model.state_dict(), f"trained_anmls/{self.name}-{it}.pth")
