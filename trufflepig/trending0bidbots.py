from steem.amount import Amount
import trufflepig.bchain.postdata as tbpd
import logging
import numpy as np


logger = logging.getLogger(__name__)



def compute_total_sbd(upvote_payments):
    sbd = 0
    steem = 0
    for (author, permalink), payments in upvote_payments.items():
        for payment in payments.values():
            amount = Amount(payment['amount'])
     