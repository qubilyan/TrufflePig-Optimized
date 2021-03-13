
import logging
import multiprocessing as mp

from steem import Steem
from steem.post import Post

from trufflepig.utils import progressbar, error_retry, none_error_retry
import trufflepig.bchain.getdata as tpbg


logger = logging.getLogger(__name__)


def extract_comment_authors_and_permalinks(operations, account):
    """Takes a list of ops and returns a set of author and permalink tuples"""
    authors_and_permalinks = []
    for operation in operations:
        try:
            op = operation['op']
            if op[0] == 'comment':
                body = op[1]['body']
                if body.startswith('@' + account):
                    comment_author = op[1]['author']
                    comment_permalink = op[1]['permlink']

                    authors_and_permalinks.append((comment_author,
                                                   comment_permalink))
        except Exception:
            logger.exception('Could not scan operation {}'.format(operation))
    return authors_and_permalinks


def check_all_ops_in_block(block_num, steem, account):
    """ Gets all posts from one block

    Parameters
    ----------
    block_num: int
    steem: Steem
    account str

    Returns
    -------
    List of tuples with comment authors and permalinks

    """
    try:
        operations = none_error_retry(steem.get_ops_in_block)(block_num, False)
        if operations:
            return extract_comment_authors_and_permalinks(operations, account)
        else:
            logger.warning('Could not find any operations for block {}'.format(block_num))
        return []
    except Exception as e:
        logger.exception('Error for block {}. Reconnecting...'.format(block_num))
        steem.reconnect()
    return []


def check_all_ops_between(start_datetime, end_datetime, steem,
                            account, stop_after=None):
    """ Queries all posts found in blocks between start and end

    Parameters
    ----------
    start_datetime: datetime
    end_datetime: datetime
    steem: Steem
    account: str
    stop_after: int or None
        For debugging

    Returns
    -------
    List of dicts of posts

    """
    start_num, block_start_datetime = tpbg.find_nearest_block_num(start_datetime, steem)
    end_num, block_end_datetime = tpbg.find_nearest_block_num(end_datetime, steem)

    total = end_num - start_num
    comment_authors_and_permalinks = []
    logger.info('Checking all operations for account {}  between '
                '{} (block {}) and {} (block {})'.format(account,
                                                         block_start_datetime,
                                                         start_num,
                                                         block_end_datetime,
                                                         end_num))

    for idx, block_num in enumerate(range(start_num, end_num+1)):
        authors_and_permalinks = check_all_ops_in_block(block_num, steem, account)
        comment_authors_and_permalinks.extend(authors_and_permalinks)
        if progressbar(idx, total, percentage_step=1, logger=logger):