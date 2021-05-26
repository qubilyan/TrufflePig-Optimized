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
                            retries=3)('@{}/{}'.format(author, permalink), steem)
        except PostDoesNotExist:
            # This happens to oftern we will suppress this
            logger.debug('Post {} by {} does not exist!'.format(permalink,
                                                                author))
            continue
        except Exception:
            logger.exception('Error in loading post {} by {}. '
                             'Reconnecting...'.format(permalink, author))
            steem.reconnect()
            continue

        # Add positive votes and subtract negative
        votes = sum(1 if x['percent'] > 0 else -1 for x in p.active_votes)

        post = {
            'title': p.title,
            'reward': p.reward.amount,
            'votes':votes,
            'active_votes': p.active_votes,
            'created': p.created,
            'tags': p.tags,
            'body': p.body,
            'author': author,
            'permalink': permalink,
            'author_reputation': int(p.author_reputation)
        }
        posts.append(post)
    return posts


def get_all_posts_from_block(block_num, steem,
                             exclude_authors_and_permalinks=None):
    """ Gets all posts from one block

    Parameters
    ----------
    block_num: int
    steem: MPSteem
    exclude_authors_and_permalinks: set of tuples of strings
        Exclude these authors and permalinks to get less duplicates

    Returns
    -------
    List of post dicts and set of authors and permalinks

    """
    try:
        operations = none_error_retry(steem.get_ops_in_block)(block_num, False)
        if operations:
            authors_and_permalinks = extract_authors_and_permalinks(operations)
            if exclude_authors_and_permalinks:
                authors_and_permalinks -= exclude_authors_and_permalinks
            if authors_and_permalinks:
                return get_post_data(authors_and_permalinks, steem), authors_and_permalinks
            else:
                logger.debug('Could not find any posts for block {}'.format(block_num))
        else:
            logger.warning('Could not find any operations for block {}'.format(block_num))
    except Exception as e:
        logger.exception('Error for block {}. Reconnecting...'.format(block_num))
        steem.reconnect()
    return [], set()


def get_all_posts_between(start_datetime, end_datetime, steem,
                          stop_after=None):
    """ Queries all posts found in blocks between start and end

    Parameters
    ----------
    start_datetime: datetime
    end_datetime: datetime
    steem: Steem
    stop_after: int or None
        For debugging and shorter tests, stop after only a few iterations

    Returns
    -------
    List of dicts of posts

    """
    start_num, block_start_datetime = find_nearest_block_num(start_datetime, steem)
    end_num, block_end_datetime = find_nearest_block_num(end_datetime, steem)

    total = end_num - start_num
    posts = []
    logger.info('Querying all posts between '
                '{} (block {}) and {} (block {})'.format(block_start_datetime,
                                                         start_num,
                                                         block_end_datetime,
                                                         end_num))
    exclude_authors_and_permalinks = set()
    for idx, block_num in enumerate(range(start_num, end_num+1)):
        posts_in_block, authors_and_permalinks = get_all_posts_from_block(block_num,
                                                                          steem,
                                                                          exclude_authors_and_permalinks)
        exclude_authors_and_permalinks |= authors_and_permalinks
        posts.extend(posts_in_block)
        if progressbar(idx, total, percentage_step=1, logger=logger):
            logger.info('Finished block {} '
                    '(last is {}) found so far {} '
                    'posts...'.format(block_num, end_num, len(posts)))
        if stop_after is not None and len(posts) >= stop_after:
            break

    logger.info('Scraped {} posts'.format(len(posts)))
    return posts


def config_mp_logging(level=logging.INFO):
    """Helper function to log in multiproc environment"""
    logging.basicConfig(level=level)


def _get_all_posts_for_blocks_parallel(block_nums, steem,
                                       stop_after=None):
    """Helper wrapper for multiprocessing"""
    posts = []
    exclude_authors_and_permalinks = set()
    for block_num in block_nums:
        posts_in_block, authors_and_permalinks = get_all_posts_from_block(block_num,
                                                           