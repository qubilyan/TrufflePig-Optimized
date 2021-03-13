
import logging
import multiprocessing as mp
import time

import pandas as pd
import numpy as np
from steem.account import Account

import trufflepig.bchain.getdata as tpbg
from trufflepig.utils import progressbar, none_error_retry


logger = logging.getLogger(__name__)


MEMO_START = 'https://steemit.com/'

BITBOTS = list({'smartmarket', 'smartsteem', 'upme', 'randowhale',
            'minnowbooster', 'boomerang', 'booster', 'hak4life',
            'lays', 'speedvoter', 'ebargains', 'danzy', 'bumper',
            'upvotewhale', 'treeplanter', 'minnowpond', 'morwhale',
            'drotto', 'postdoctor', 'moonbot', 'tipu', 'blockgators',
            'echowhale', 'steemvote', 'byresteem', 'originalworks', 'withsmn',
            'siditech', 'alphaprime', 'hugewhale', 'steemvoter', 'hottopic',
            'resteemable', 'earthnation-bot', 'photocontests', 'friends-bot',
           'followforupvotes', 'frontrunner', 'resteembot', 'steemlike',
           'thundercurator', 'earnmoresteem', 'microbot', 'coolbot',
           'thehumanbot', 'steemthat', 'gangvote', 'refresh', 'cabbage-dealer',
           'growingpower', 'postresteem', 'mecurator', 'talhadogan',
           'okankarol', 'bidseption', 'highvote', 'oguzhangazi', 'ottoman',
           'resteemr', 'superbot', 'bestvote', 'zerotoherobot', 'red-rose',
           'jeryalex', 'oceansbot', 'fresteem', 'otobot', 'bidbot',
           'honestbot', 'upgoater', 'whalebuilder', 'postpromoter', 'pwrup',
           'spydo', 'upmewhale', 'promobot', 'puppybot', 'moneymatchgaming',
           'sneaky-ninja', 'zapzap', 'sleeplesswhale', 'estream.studios',
           'seakraken', 'canalcrypto', 'upmyvote', 'hotbot',
           'redlambo', 'slimwhale', 'singing.beauty', 'inciter', 'lovejuice',
           'steembidbot', 'bid4joy', 'mitsuko', 'pushup', 'luckyvotes',
           'discordia', 'shares', 'postdoctor', 'upboater',
           'megabot', 'dailyupvotes', 'ebargains', 'bluebot', 'upyou',
           'edensgarden', 'smartwhale', 'voterunner', 'nado.bot',
           'jerrybanfield', 'foxyd', 'onlyprofitbot', 'minnowhelper',
           'msp-bidbot', 'therising', 'bearwards', 'thebot', 'buildawhale',
           'chronocrypto', 'brupvoter', 'smartsteem', 'payforplay',
           'adriatik', 'cryptoempire', 'isotonic', 'minnowfairy',
           'appreciator', 'childfund', 'mercurybot', 'allaz', 'sunrawhale',
           'mrswhale', 'kittybot', 'lightningbolt', 'hottopic',
           'sportic'})


def find_nearest_index(target_datetime,
                           account,
                           steem,
                           latest_index=None,
                           max_tries=5000,
                           index_tolerance=2):
    """ Finds nearest account action index to `target_datetime`

    Parameters
    ----------
    target_datetime: datetime
    steem: Steem object
    latest_index: int
        latest index number in acount index
        leave None to get from steem directly
    max_tries: int
        number of maximum tries
    index_tolerance: int
        tolerance too closest index number

    Returns
    -------
    int: best matching index
    datetime: datetime of matching index

    """
    acc = none_error_retry(Account,
                           errors=(Exception,))(account, steem)

    if latest_index is None:
        latest_index = none_error_retry(next, errors=(Exception,))(none_error_retry(acc.history_reverse,
                                   errors=(Exception,))(batch_size=1))['index']

    current_index = latest_index
    best_largest_index = latest_index

    action = none_error_retry(next, errors=(Exception,))(none_error_retry(acc.get_account_history,
                                   errors=(Exception,))(best_largest_index, limit=1))
    best_largest_datetime = pd.to_datetime(action['timestamp'])
    if target_datetime > best_largest_datetime:
        logger.debug('Target beyond largest block num')
        return latest_index, best_largest_datetime

    best_smallest_index = 1
    increase = index_tolerance + 1
    current_datetime = None
    for _ in range(max_tries):
        try:
            action = none_error_retry(next, errors=(Exception,))(none_error_retry(acc.get_account_history,
                                   errors=(Exception,))(current_index, limit=1))
            current_datetime = pd.to_datetime(action['timestamp'])
            if increase <= index_tolerance:
                return current_index, current_datetime
            else:

                if current_datetime < target_datetime:
                    best_smallest_index = current_index
                else:
                    best_largest_index = current_index

                increase = (best_largest_index - best_smallest_index) // 2
                current_index = best_smallest_index + increase

                if current_index < 0 or current_index > latest_index:
                    raise RuntimeError('Seriously? Error for '
                                       'account {}: current_index {} '
                                       'latest_index {}'.format(account,
                                                                current_index,
                                                                latest_index))
        except StopIteration:
            logger.exception('Problems for index {} of account {}. '
                             'Reraising...'.format(current_index, account))
            raise
        except Exception:
            logger.exception('Problems for index {} of account {}. '
                             'Reconnecting...'.format(current_index, account))
            current_index -= 1
            best_largest_index -= 1
            steem.reconnect()
            acc = none_error_retry(Account,
                                   errors=(Exception,))(account, steem)
            if current_index <= 1:
                logger.error('Could not find index, raising StopIteration')
                raise StopIteration('Problems for account {}'.format(account))


def get_delegates_and_shares(account, steem):
    """ Queries all delegators to `account` and the amount of shares

    Parameters
    ----------
    account: str
    steem: Steem

    Returns
    -------
    dict of float

    """
    acc = none_error_retry(Account,
                           errors=(Exception,))(account, steem)
    delegators = {}
    for tr in none_error_retry(acc.history_reverse,
                                   errors=(Exception,))(filter_by='delegate_vesting_shares'):
        try:
            delegator = tr['delegator']
            if delegator not in delegators:
                shares = tr['vesting_shares']
                if shares.endswith(' VESTS'):
                    shares = float(shares[:-6])
                    timestamp = pd.to_datetime(tr['timestamp'])
                    delegators[delegator] = {'vests': shares,
                                             'timestamp': timestamp}
                else:
                    raise RuntimeError('Weird shares {}'.format(shares))

        except Exception as e:
            logger.exception('Error extracting delegator from '
                             '{}, restarting steem'.format(tr))
            steem.reconnect()
    return delegators


def get_delegate_payouts(account, steem, current_datetime,
                         min_days, investor_share):
    """ Returns pending payouts for investors

    Parameters
    ----------
    account: str
    steem: Steem
    current_datetime: datetime
    min_days: int
        minimum days of delegation before payout
    investor_share: float

    Returns
    -------
    dict of float:
        SBD to pay to each investor
    dict of float:
        STEEM to pay to each investor

    """
    assert 0 < investor_share <= 1

    current_datetime = pd.to_datetime(current_datetime)
    threshold_date = current_datetime - pd.Timedelta(days=min_days)

    vests_by = none_error_retry(get_delegates_and_shares,
                                retries=3, errors=(TypeError,))(account, steem)
    filtered_vests_by = {delegator: dict_['vests']
                         for delegator, dict_ in vests_by.items()
                            if dict_['timestamp'] < threshold_date}
    acc = none_error_retry(Account,
                           errors=(Exception,))(account, steem)

    pending_sbd = acc.balances['rewards']['SBD']
    pending_steem = acc.balances['rewards']['STEEM']
    vests = acc.balances['total']['VESTS']
    filtered_vests_by[account] = vests

    total_vests = sum(filtered_vests_by.values())
    sbd_payouts = {delegator: np.round(vests / total_vests * investor_share * pending_sbd, decimals=3)
                    for delegator, vests in filtered_vests_by.items() if delegator != account}

    steem_payouts = {delegator: np.round(vests / total_vests * investor_share * pending_steem, decimals=3)
                    for delegator, vests in filtered_vests_by.items() if delegator != account}

    return sbd_payouts, steem_payouts


def get_upvote_payments(account, steem, min_datetime, max_datetime,
                        batch_size=1000, max_time=1800):

    start = time.time()
    upvote_payments = {}

    try:
        start_index, _ = find_nearest_index(max_datetime,
                                         account, steem)
    except StopIteration:
        logger.exception('Could not get account INDEX data from '
                         '{}. Reconnecting'.format(account))
        steem.reconnect()
        start_index = 1

    try:
        transfers = history_reverse(account, steem, filter_by='transfer',
                                    start_index=start_index,
                                    batch_size=batch_size)
    except Exception as e:
        logger.exception('Could not get account data from '
                         '{}, restarting steem'.format(account))
        transfers = []
        steem.reconnect()