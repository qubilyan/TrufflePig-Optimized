import argparse
import concurrent
import gc
import logging
import os
import time

import pandas as pd

import trufflepig.bchain.getdata as tpgd
import trufflepig.bchain.postdata as tppd
import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.utils as tfut
import trufflepig.pigonduty as tfod
import trufflepig.bchain.paydelegates as tpde
import trufflepig.bchain.getaccountdata as tpad
from trufflepig import config
from trufflepig.utils import configure_logging
import trufflepig.bchain.postweeklyupdate as tppw
from trufflepig.bchain.mpsteem import MPSteem
from trufflepig.bchain.poster import Poster
import trufflepig.trending0bidbots as tt0b


logger = logging.getLogger(__name__)


MAX_DOCUMENTS = 123000


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='TrufflePig Bot')
    parser.add_argument('--broadcast', action="store_false",
                        default=True)
    parser.add_argument('--now', action='store', default=None)
    args = parser.parse_args()
    return args.broadcast, args.now


def large_mp_preprocess(log_directory, current_datetime, steem, data_directory,
                        days, offset_days):
    """Helper function to spawn in child process"""
    configure_logging(log_directory, current_datetime)
    post_frame = tpgd.load_or_scrape_training_data(steem, data_directory,
                                                       current_datetime=current_datetime,
                                                       days=days,
                                     