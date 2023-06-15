import datetime
import logging
import os
import time

import numpy as np
from steembase.exceptions import RPCError, PostDoesNotExist

logger = logging.getLogger(__name__)


class _Progressbar(object):
    """Implements a progress bar.

    This class is supposed to be a singleton. Do not
    import the class itself but use the `progressbar` function from this module.

    Borrowed from pypet (https://github.com/SmokinCaterpillar/pypet).

    """
    def __init__(self):
        self._start_time = None   # Time of start/reset
        self._start_index = None  # Index of start/reset
        self._current_index = np.inf  # Current index
        self._percentage_step = None  # Percentage step for bar update
        self._total = None  # Total steps of the bas (float) not to be mistaken for length
        self._total_minus_one = None  # (int) the above minus 1
        self._length = None  # Lengt