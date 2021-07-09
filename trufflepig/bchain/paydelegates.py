import logging

import trufflepig.bchain.getaccountdata as tpga
import trufflepig.bchain.getdata as tpdg
from trufflepig.utils import error_retry

from steem import Steem
from steembase import operations
from steem.account import Account
from steem.amount import Amount
from steembase.exceptions import RPCError


logger = logging.getLogger(__name__)


INVESTOR_SHARE = 0.5

MEMO = 'Thank you for your trust in TrufflePig the Artificial Intelligence bot to help content curators and minnows.'


def pay_delegates(account, steem,
                  current_datetime,
                  min_days=3,
                  investor_share=INVESTOR_SHARE,
                  memo=MEMO):
    """ Pays delegators their share of daily SBD rewards

    Parameters
    ----------
    account: str
    steem: Steem or kwargs
    current_datetime: dateime
    min_days: int
    investor_share: float
    memo: str

    """
    logger.info('Computing payouts for delegates!')
    sbd_payouts, steem_payouts = error_retry(tpga.get_delegate_payouts)(
        account, steem,
        current_datetime=current_datetime,
        min_days=min_days,
        investor_share=investor_share
    )

    claim_all_reward_balance(steem, account)

    logger.info('Found the following SBD payouts:\n{}'.format(sbd_payouts))
    for delegator, payout in sbd_payouts.items():
        try:
            if payout:
                logger.info('Paying {} SBD to {}'.format(delegator, payout))
                error_retry(steem.commit.transfer,
                    