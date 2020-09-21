import logging
import os
import gc

import pandas as pd
from steem import Steem

import trufflepig.model as tpmo
import trufflepig.preprocessing as tppp
import trufflepig.bchain.getdata as tpgd
from trufflepig.bchain.mpsteem import MPSteem
from trufflepig import config


def main():

    format=('%(asctime)s %(processName)s:%(name)s:'
                  '%(funcName)s:%(lineno)s:%(levelname)s: %(message)s')
    logging.basicConfig(level=logging.INFO, format=format)
    directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')

    steem = MPSteem(nodes=config.NODES, no_broadcast=True)
    current_datetime = pd.to_datetime('2018-02-01')

    crossval_filename = os.path.join(directory, 'xval_{}.gz'.format(current_datetime.date()))

    post_frame = tpgd.load_or_scrape_training_data(steem, directory,
                                                   current_datetime=current_datetime,
                                                   days=10,
                                                   offset_days=0)

    gc.collect()

    regressor_kwargs = dict(n_estimator