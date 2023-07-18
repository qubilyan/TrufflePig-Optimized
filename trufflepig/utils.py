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
                total_seconds = time_delta.total_seconds()
            except AttributeError:
                # for backwards-compatibility
                # Python 2.6 does not support `total_seconds`
                total_seconds = ((time_delta.microseconds +
                                    (time_delta.seconds +
                                     time_delta.days * 24 * 3600) * 10 ** 6) / 10.0 ** 6)
            remaining_seconds = int((self._total - self._start_index - 1.0) *
                                    total_seconds / float(index - self._start_index) -
                                    total_seconds)
            remaining_delta = datetime.timedelta(seconds=remaining_seconds)
            remaining_str = ', remaining: ' + str(remaining_delta)
        except ZeroDivisionError:
            remaining_str = ''
        return remaining_str

    def __call__(self, index, total, percentage_step=5, logger='print', log_level=logging.INFO,
                 reprint=False, time=True, length=20, fmt_string=None,  reset=False):
        """Plots a progress bar to the given `logger` for large for loops.

        To be used inside a for-loop at the end of the loop.

        :param index: Current index of for-loop
        :param total: Total size of for-loop
        :param percentage_step: Percentage step with which the bar should be updated
        :param logger:

            Logger to write to, if string 'print' is given, the print statement is
            used. Use None if you don't want to print or log the progressbar statement.

        :param log_level: Log level with which to log.
        :param reprint:

            If no new line should be plotted but carriage return (works only for printing)

        :param time: If the remaining time should be calculated and displayed
        :param length: Length of the bar in `=` signs.
        :param fmt_string:

            A string which contains exactly one `%s` in order to incorporate the progressbar.
            If such a string is given, ``fmt_string % progressbar`` is printed/logged.

        :param reset:

            If the progressbar should be restarted. If progressbar is called with a lower
            index than the one before, the progressbar is automatically restarted.

        :return:

            The progressbar string or None if the string has not been updated.


        """
        reset = (reset or
                 index <= self._current_index or
                 total != self._total)
        if reset:
            self._reset(index, total, percentage_step, length)

        statement = None
        indexp1 = index + 1.0
        next_interval = int(indexp1 / self._norm_factor)
        ending = index >