import logging
import os
import multiprocessing as mp
import gc

import pandas as pd
import numpy as np
import scipy.stats as spst
from steem.amount import Amount

import trufflepig.filters.stylemeasures as tfsm
import trufflepig.filters.textfilters as tftf
import trufflepig.bchain.getaccountdata as tfga
from trufflepig.filters.blacklist import BUILD_A_WHALE_BLACKLIST


logger = logging.getLogger(__name__)


FILTER_TAGS = ('mitnebcurationtrail', 'informationwar', 'truth', 'conspiracy',
               'vaccines', 'contest', 'giveaway', 'deutsch', 'kr', 'kr-newbie',
               'nsfw', 'sex', 'daily', 'photofeed', 'gambling',
               # other weird stuff
               'steemsilvergold', 'horoscope', 'guns', 'investing', 'tib',
               # Somehow religious texts do not work in combination with others
               # maybe I need a bot just to rate spiritual content
               # for simplicity let's ignore them for now,
               # sorry, no releigious truffles in the near future!
               'bible', 'faith', 'spiritual', 'christianity', 'steemchurch',
               # Filter translations for utoptian
               'translations', 'translation')


# Stay out of the whale wars!
FILTER_AUTHORS = ('haejin', 'ew-and-patterns', 'caladium',
                  'cryptopassion', 'thirdeye7', 'shariarahammad')

# Get out plagiarismos!
FILTER_VOTERS = ('cheetah',)


def filter_duplicates(frame):
    """ Filters out duplicate entries based on author and permalink

    Filtering is inplace!

    Parameters
    ----------
    frame: DataFrame

    Returns
    -------
    DataFrame

    """
    old_len = len(frame)
    frame.drop_duplicates(subset=['author', 'permalink'],
                                     keep='last', inplace=True)
    if len(frame) < old_len:
        logger.info('Filtered {} duplicates kept {} '
                    'posts'.format(old_len - len(frame), len(frame)))
    return frame


def apply_parallel(function, iterable, ncores, chunksize=1000):
    """ Applies a `function` in parallel on `ncores`.

    Parameters
    ----------
    function: callable
    iterable: list, tuple, etc.
    ncores: int
        The number of jobs started
    chunksize: int
        Size of chunk submitted to pool

    Returns
    -------
    List of function outputs

    """
    if ncores == 1:
        return [function(x) for x in iterable]
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(ncores)