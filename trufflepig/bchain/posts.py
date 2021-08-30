import numpy as np

import trufflepig.filters.textfilters as tftf


TRUFFLE_IMAGE_SMALL = '![trufflepig](https://raw.githubusercontent.com/SmokinCaterpillar/TrufflePig/master/img/trufflepig17_small.png)'
TRUFFLE_IMAGE = '![trufflepig](https://raw.githubusercontent.com/SmokinCaterpillar/TrufflePig/master/img/trufflepig17.png)'
DELEGATION_LINK = 'https://v2.steemconnect.com/sign/delegateVestingShares?delegator=&delegatee=trufflepig&vesting_shares={shares}%20VESTS'
QUOTE_MAX_LENGTH = 496
TAGS = ['steemit', 'curation', 'minnowsupport', 'technology', 'community']
TRENDING_TAGS = ['steemit', 'curation', 'bots', 'technology', 'community']

BODY_PREFIX = ''  # to announce tests etc.


def truffle_comment(reward, votes, rank, topN_link, truffle_link, truffle_image_small=TRUFFLE_IMAGE_SMALL):
    """Creates a comment made under an upvoted toplist post"""
    post = """**Congratulations!** Your post has been selected as a daily Steemit truffle! It is listed on **rank {rank}** of all contributions awarded today. You can find the [TOP DAILY TRUFFLE PICKS HERE.]({topN_link}) 
    
I upvoted your contribution because to my mind your post is at least **{reward} SBD** worth and should receive **{votes} votes**. It's now up to the lovely Steemit community to make this come