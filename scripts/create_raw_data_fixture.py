import os
import logging

from trufflepig import config
import trufflepig.bchain.getdata as tpbg

def main():

    logging.basicConfig(level=logging.INFO)
    directory = os.path.join(config.PROJECT_DIRECTORY, 'scraped_data')


    frames = tpbg.load_or_scrape_training_data(dict(nodes=config.NODES),
                                               directory,
                                              