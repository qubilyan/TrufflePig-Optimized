import logging
import os
import multiprocessing as mp
from collections import OrderedDict

import pandas as pd
from steem import Steem
from steem.blockchain import Blockchain
from steem.post import Post, PostDoesNotExist
import json
from json import JSONDecodeError

from trufflepig.utils import progressbar, error_retry, none_error_retry
import trufflepig.persist as tppe


logger = logging.getLogger(__name__)


MIN_CHARACTERS = 500

FILENAME_TEMPLATE = 'steemit_posts__{time}.sqlite'

TABLENAME = 'steemit_posts'


################################### Block Utils #################################


def get_block_headers_between_offset_start(start_datetime, end_datetime,
                                           end_offset_num, steem):
    """ Returns block headers between a date range

    NOT used in production!

    Parameters
    ----------
    start_datetime: datetime
    end_datetime: datetime
    end_offset_num: offset from wich to seach backwards
    steem: Steem object

    Returns
    -------
    Ordereddict: block_num -> header

    """
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    current_block_num = end_offset_num
    headers = OrderedDict()
    logger.info('Collecting header infos')
    while True:
        try:
            header = none_error_retry(steem.get_block_header)(current_block_num)
            current_datetime = pd.to_datetime(header['timestamp'])
            if start_datetime <= current_datetime and current_datetime <= end_datetime:
                header['timestamp'] = current_datetime
                headers[current_block_num] = header
            if current_datetime < start_datetime:
                break
        except Exception:
            logger.exception('Error for block num {}. Reconnecting...'.format(current_block_num))
            steem.reconnect()
        current_block_num -= 1
        if current_block_num % 100 == 99:
            logger.debug('Bin alread {} headers'.format(len(headers)))
    return headers


def find_nearest_block_num(target_datetime, steem,
                           latest_block_num=None,
                           max_tries=5000,
                           block_num_tolerance=0):
    """ Finds nearest block number to `target_datetime`

    Parameters
    ----------
    target_datetime: datetime
    steem: Steem object
    latest_block_num: int
        latest block number in bchain
        leave None to get from steem directly
    max_tries: int
        number of maximum tries
    block_num_tolerance: int
        tolerance too closest in block

    Returns
    -------
    int: best matching block number
    datetime: datetime of matching block

    """
    if latest_block_num is None:
        latest_block_num = none_error_retry(Blockchain(steem).get_current_block_num)()

    current_block_num = latest_block_num
    best_largest_block_num = latest_block_num

    header = none_error_retry(steem.get_block_header)(best_largest_block_num)
    best_largest_datetime = pd.to_datetime(header['timestamp'])
    if target_datetime > best_largest_datetime:
        logger.warning('Target beyond largest block num')
        return latest_block_num, best_largest_datetime

    best_smallest_block_num = 1
    increase = block_num_tolerance + 1
    current_datetime = None
    for _ in range(max_tries):
        try:
            header = none_error_retry(steem.get_block_header)(current_block_num)
            current_datetime = pd.to_datetime(header['timestamp'])
            if increase <= block_num_tolerance:
                return current_block_num, current_datetime
            else:

                if current_datetime < target_datetime:
                    best_smallest_block_num = current_block_num
                else:
                    best_largest_block_num = current_block_num

                increase = (best_largest_block_num - best_smallest_block_num) // 2
                current_block_num = best_smallest_block_num + increase

                if current_block_num < 0 or current_block_num > latest_block_num:
                    raise RuntimeError('Seriously?')
        except Exception:
            logger.exception('Problems for block num {}. Reconnecting...'
                             ''.format(current_block_num))
            current_block_num -= 1
            best_smallest_block_num -= 1
            steem.reconnect()
            if current_block_num <= 1:
                logger.error('Could not find block num returning 1')
                return 1, current_datetime


def get_block_headers_between(start_datetime, end_datetime, steem):
    """ Returns block headers between two dates"""
    latest_block_num = Blockchain(steem).get_current_block_num()
    end_offset_num, _ = find_nearest_block_num(end_datetime, steem, latest_block_num)
    return get_block_headers_between_offset_start(start_datetime, end_datetime,
                                                  steem=steem,
                                                  end_offset_num=end_offset_num)


################################### Post Data #################################


def extract_authors_and_permalinks(operations):
    """Takes a list of ops and returns a set of author and permalink tuples"""
    authors_and_permalinks = set()
    for operation in operations:
        op = operation['op']
        if op[0] == 'comment':
            title = op[1]['title']
            body = op[1]['body']
            if title != '' and op[1]['json_metadata'] != '' and len(body) >= MIN_CHARACTERS:
                try:
                    metadata = json.loads(op[1]['json_metadata'])
                except JSONDecodeError:
                    logger.debug('Could not decode metadata for {}'.format(op))
                    continue
                try:
                    tags = metadata['tags']
                except KeyError as e:
                    logger.debug('No tags for for {}'.format(op))
                    continue
                except TypeError as e:
                    logger.debug('Type Error for for {}'.format(op))
                    continue
                try:
                    _ = tags[0]
                except IndexError as e:
                    logger.debug('Tags empty for {}'.format(op))
                    continue
                author = op[1]['author']
                permalink = op[1]['permlink']
                authors_and_permalinks.add((author, permalink))
    return authors_and_permalinks


def get_post_data(authors_and_permalinks, steem):
    """ Queries posts from `steem`

    Parameters
    ----------
    authors_and_permalinks: set of tuples of authors and permalink strings
    steem: Steem object

    Returns
    -------
    List of dict
        posts are kept as dicts with
            * author
            * permalink
            * title
            * body
            * reward
            * votes
            * created
            * tags

    """
    posts = []
    for kdx, (author, permalink) in enumerate(authors_and_permalinks):
        try:
            p = error_retry(Post,
                            errors=Exception,
                            sleep_time=0.5,
                            retries=3)('@{}/