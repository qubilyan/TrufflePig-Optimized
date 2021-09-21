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
    
I upvoted your contribution because to my mind your post is at least **{reward} SBD** worth and should receive **{votes} votes**. It's now up to the lovely Steemit community to make this come true.

I am `TrufflePig`, an Artificial Intelligence Bot that helps minnows and content curators using Machine Learning. If you are curious how I select content, [you can find an explanation here!]({truffle_link})
    
Have a nice day and sincerely yours,
{truffle_image_small}
*`TrufflePig`*
    """
    post = BODY_PREFIX + post

    return post.format(reward=int(reward), votes=int(votes), topN_link=topN_link,
                       truffle_link=truffle_link, rank=rank,
                       truffle_image_small=truffle_image_small)


def topN_list(topN_authors, topN_permalinks, topN_titles,
              topN_filtered_bodies, topN_image_urls,
              topN_rewards, topN_votes, quote_max_length, nstart=1):
    """Creates a toplist string"""
    topN_entry="""**#{rank}** [{title}](https://steemit.com/@{author}/{permalink})  --  **by @{author} with an estimated worth of {reward:d} SBD and {votes:d} votes**
    
{image}{quote}

"""

    result_string = ""

    iterable = zip(topN_authors, topN_permalinks, topN_titles,
                   topN_filtered_bodies, topN_image_urls,
                   topN_rewards, topN_votes)

    for idx, (author, permalink, title, filtered_body, img_urls, reward, votes) in enumerate(iterable):
        rank = idx + nstart
        quote = '>' + filtered_body[:quote_max_length].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') + '...'
        title = tftf.replace_newlines(title)
        title = tftf.filter_special_characters(title)
        if len(img_urls) >= 1:
            imgstr = """ <div class="pull-right"><img src="{img}" /></div>\n\n""".format(img=img_urls[0])
        else:
            imgstr=''
        entry = topN_entry.format(rank=rank, author=author, permalink=permalink,
                                   title=title, quote=quote, votes=int(votes),
                                   reward=int(reward), image=imgstr)
        result_string += entry
    return result_string


def simple_topN_list(topN_authors, topN_permalinks, topN_titles,
                     topN_rewards, topN_votes, nstart):
    """Creates a toplist for lower ranks"""
    topN_entry="""\n {rank}: [{title}](https://steemit.com/@{author}/{permalink}) (by @{author}, {reward:d} SBD, {votes:d} votes)\n"""

    result_string = ""

    iterable = zip(topN_authors, topN_permalinks, topN_titles,
                   topN_rewards, topN_votes)

    for idx, (author, permalink, title, reward, votes) in enumerate(iterable):
        rank = idx + nstart
        title = tftf.replace_newlines(title)
        title = tftf.filter_special_characters(title)
        entry = topN_entry.format(rank=rank, author=author, permalink=permalink,
                                   title=title, votes=int(votes),
                                   reward=int(reward))
        result_string += entry
    return result_string


def get_delegation_link(steem_per_mvests, steem_powers=(2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)):
    """Returns a dictionary of links to delegate SP"""
    link_dict = {}
    for steem_power in steem_powers:
        shares = np.round(steem_power / steem_per_mvests * 1e6, 3)
        link_dict['sp'+str(steem_power)] = DELEGATION_LINK.format(shares=shares)
    return link_dict


def topN_post(topN_authors, topN_permalinks, topN_titles, topN_filtered_bodies,
              topN_image_urls, topN_rewards, topN_votes, title_date,
              truffle_link, steem_per_mvests=490, truffle_image=TRUFFLE_IMAGE,
              quote_max_length=QUOTE_MAX_LENGTH):
    """Craetes the truffle pig daily toplist post"""
    title = """Today's Truffle Picks: Quality Steemit Posts that deserve more Rewards and Attention! ({date})"""

    post=""" ## Daily Truffle Picks
    
It's time for another round of truffles I found digging in the streams of this beautiful platform!

For those of you who do not know me: My name is *TrufflePig*. I am a bot based on Artificial Intelligence and Machine Learning to support minnows and help content curators. I was created and am being maintained by @smcaterpillar. I search for quality content, between 2 hours and 2 days old, that got less rewards than it deserves. I call these posts truffles, publish a daily top list, an