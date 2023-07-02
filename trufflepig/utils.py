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
        self._length = None  # Length of the percentage bar in `=` signs
        self._norm_factor = None  # Normalization factor
        self._current_interval = None  # The current interval,
        # to check if bar needs to be updated

    def _reset(self, index, total, percentage_step, length):
        """Resets to the progressbar to start a new one"""
        self._start_time = datetime.datetime.now()
        self._start_index = index
        self._current_index = index
        self._percentage_step = percentage_step
        self._total = float(total)
        self._total_minus_one = total - 1
        self._length = length
        self._norm_factor = max(total * percentage_step / 100.0, 1)
        self._current_interval = int((index + 1.0) / self._norm_factor)

    def _get_remaining(self, index):
        """Calculates remaining time as a string"""
        try:
            current_time = datetime.datetime.now()
            time_delta = current_time - self._start_time
            try:
                total_seconds = tim