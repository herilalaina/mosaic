"""Utils file."""

import random
import math
import signal
import numpy as np

def random_uniform_on_log_space(a, b):
    return math.exp(random.uniform(a, b))


def get_index_percentile(vect, perc):
    if len(vect) == 1:
        return 0
    try:
        idx = perc * (len(vect) - 1)
        idx = math.ceil(idx + 0.5)
        return np.argpartition(vect, idx)[idx]
    except:
        return np.argmax(vect)


class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()
