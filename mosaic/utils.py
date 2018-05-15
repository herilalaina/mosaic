"""Utils file."""

import random
import math

def random_uniform_on_log_space(a, b):
    return math.exp(random.uniform(a, b))
