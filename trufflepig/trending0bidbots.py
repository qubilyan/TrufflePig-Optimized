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
            value = amount.amount
            asset = amount.asset
            if asset == 'SBD':
                sbd += value
            elif asset == 'STEEM':
                steem += value
    return sbd, steem


def create_trending_post(post_frame, upvote_payments, poster, topN_permalink,
                         overview_permalink, current_datetime, bots=()):
    total_paid_sbd, total_paid_steem = compute_total_sbd(upvote_payments)

    bots = set(bots)

    logger.info('People spend {} SBD and {} Steem on Bid Bots the last 24 '
                'hours.'.format(total_paid_sbd, total_paid_steem))

    # exclude bit bots
    no_bid_bots_frame = post_frame.loc[post_frame.bought_votes == 0, :].copy()

    # exclude self votes
    self_votes = []
    for idx, row in no_bid_bots_frame.iterrows():
        self_votes.append(row.author in {x['voter'] for x in row.active_votes})
    self_votes = np.array(self_votes)
    no_bid_bots_frame = no_bid_bots_frame.loc[~self_votes, :]

    # exlude all bot votes
    bot_votes = []
    for idx, row in no_bid_bots_frame.iterrows():
        bot_votes.append(len(bots.intersection(
            {x['voter'] for x in row.active_votes})
        ) > 0)
    bot_votes = np.array(bot_votes)
    no_bid_bots_frame = no_bid_bots_frame.loc[~bot_votes, :]

    no_bid_bots_fram