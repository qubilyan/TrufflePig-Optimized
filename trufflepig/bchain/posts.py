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

For those of you who do not know me: My name is *TrufflePig*. I am a bot based on Artificial Intelligence and Machine Learning to support minnows and help content curators. I was created and am being maintained by @smcaterpillar. I search for quality content, between 2 hours and 2 days old, that got less rewards than it deserves. I call these posts truffles, publish a daily top list, and upvote them. Now it is up to you to give these posts the attention they deserve. If you are curious how I select content, [you can find an explanation here.]({truffle_link})
    
Please, be aware that the list below has been automatically generated by a Machine Learning algorithm that was trained on payouts of previous contributions of the Steemit community. Of course, **this algorithm can make mistakes**. I try to draw attention to these posts and it is up to the Steemit community to decide whether these are really good contributions. Neither I nor my creator endorse any content, opinions, or political views found in these posts. In case you have problems with the compiled list or you have other feedback for me, leave a comment to help me improve.
    
# The Top 10 Truffles

Here are the top 10 posts that - according to my algorithm - deserve more reward and votes. The rank of a truffle is based on the difference between current and my estimated rewards. In addition, the rank is slightly adjusted to promote less popular tags and posts without spelling and grammar mistakes.

{topN_truffles}


### You didn't make it into the top list this time?

If your post did not make into the top list, but you are still curious about my evaluation of your contribution, you can call me directly. Just reply to your own post with @trufflepig. I will answer the call within the next 24 hours.

## You can Help and Contribute

By checking, upvoting, and resteeming the found truffles from above, you help minnows and promote good content on Steemit. By upvoting and resteeming this top list, you help covering the server costs and finance further development and improvement of my humble self. 

**NEW**: You may further show your support for me and all the found truffles by [**following my curation trail**](https://steemauto.com/dash.php?trail=trufflepig&i=1) on SteemAuto!

## Delegate and Invest in the Bot

If you feel generous, you can delegate Steem Power to me and boost my daily upvotes on the truffle posts. In return, I will provide you with a *small* compensation for your trust in me and your locked Steem Power. **Half of my daily SBD and STEEM income will be paid out to all my delegators** proportional to their Steem Power share. Payouts will start 3 days after your delegation.

Click on one of the following links to delegate **[2]({sp2}), [5]({sp5}), [10]({sp10}), [20]({sp20}), [50]({sp50}), [100]({sp100}), [200]({sp200}), [500]({sp500}), [1000]({sp1000}), [2000]({sp2000}),** or even **[5000 Steem Power]({sp5000})**. Thank You!

Cheers,

{truffle_image}

*`TrufflePig`*
    """
    link_dict = get_delegation_link(steem_per_mvests=steem_per_mvests)

    topN_truffles = topN_list(topN_authors=topN_authors,
                              topN_permalinks=topN_permalinks,
                              topN_titles=topN_titles,
                              topN_filtered_bodies=topN_filtered_bodies,
                              topN_image_urls=topN_image_urls,
                              topN_rewards=topN_rewards,
                              topN_votes=topN_votes,
                              quote_max_length=quote_max_length)

    title = title.format(date=title_date.strftime('%d.%m.%Y'))
    post = post.format(topN_truffles=topN_truffles,
                          truffle_image=truffle_image,
                          truffle_link=truffle_link,
                          **link_dict)
    post = BODY_PREFIX + post
    return title, post


def topN_comment(topN_authors, topN_permalinks, topN_titles,
                 topN_rewards, topN_votes, nstart=11):
    """Creates the toplist comment for lower ranks"""
    post = """If you cannot get enough truffles, here are ranks 11 till 25:
    
{topN_truffles}
    """

    topN_truffles = simple_topN_list(topN_authors=topN_authors,
                                     topN_permalinks=topN_permalinks,
                                     topN_titles=topN_titles,
                                     topN_votes=topN_votes,
                                     topN_rewards=topN_rewards,
                                     nstart=nstart)
    post = post.format(topN_truffles=topN_truffles)
    return post


def on_call_comment(author, reward, votes, topN_link, truffle_link, truffle_image_small=TRUFFLE_IMAGE_SMALL):
    """Creates a comment made under an upvoted toplist post"""
    post = """Thanks for calling @{author}! Here is a small upvote for this post and my opinion about it.
    
To my mind this post is at least **{reward} SBD** worth and should receive **{votes} votes**.

By the way, you can find [TODAY'S TRUFFLE PICKS HERE.]({topN_link}) 

I am `TrufflePig`, an Artificial Intelligence Bot that helps minnows and content curators using Machine Learning. If you are curious how I evaluate content, [you can find an explanation here!]({truffle_link})
    
Have a nice day and sincerely yours,
{truffle_image_small}
*`TrufflePig`*
    """
    post = BODY_PREFIX + post

    # Well, let's be nice ;-)
    reward += 1
    return post.format(author=author, reward=int(reward), votes=int(votes), topN_link=topN_link,
                       truffle_link=truffle_link,
                       truffle_image_small=truffle_image_small)


def topN_words(words, counts):
    result = '\n'
    for irun, (word, count) in enumerate(zip(words, counts)):
        result += '{rank}. {word}: {count}\n'.format(rank=irun + 1,
                                                     word=word,
                                                     count=count)
    return result


def topN_tfidf(words, tfidfs):
    result = '\n'
    for irun, (word, tfidf) in enumerate(zip(words, tfidfs)):
        result += '{rank}. {word}: {tfidf:.2f} tfidf score\n'.format(rank=irun + 1,
                                                                             word=word,
                                                                             tfidf=tfidf)
    return result


def topN_tags(tags, counts, rewards):
    result = '\n'
    for irun, (tag, count, reward) in enumerate(zip(tags, counts, rewards)):
        result += '{rank}. {tag}: {count} with {reward} SBD\n'.format(rank=irun + 1,
                                                                      tag=tag,
                                                                      count=count,
                                                                      reward=int(reward))
    return result


def topN_tags_earnings(tags, counts, rewards_per_post):
    result = '\n'
    for irun, (tag, count, reward) in enumerate(zip(tags, counts, rewards_per_post)):
        result += '{rank}. {tag}: {count} with {reward:.3f} SBD per post\n'.format(rank=irun + 1,
                                                                      tag=tag,
                                                                      count=count,
                                                                      reward=reward)
    return result


def topN_posters(authors, titles, permalinks, rewards):
    result = '\n'
    for irun, (author, title, permalink, reward) in enumerate(zip(authors, titles, permalinks, rewards)):
        result += ("{rank}. ['{title}'](https://steemit.com/@{author}/{permalink}) by @{author} worth **{reward} "
                  "SBD**\n").format(rank=irun + 1, author=author, title=title,
                                    permalink=permalink, reward=int(reward))
    return result


def weekly_update(current_datetime,
                  steem_per_mvests,
                  start_datetime,
                  end_datetime,
                  total_posts,
                  total_votes,
                  total_reward,
                  bid_bots_sbd,
                  bid_bots_steem,
                  bid_bots_percent,
                  median_reward,
                  mean_reward,
                  dollar_percent,
                  top_posts_authors,
                  top_posts_titles,
                  top_posts_rewards,
                  top_posts_permalinks,
                  top_tags,
                  top_tag_counts,
                  top_tag_rewards,
                  top_tags_earnings,
                  top_tags_earnings_counts,
                  top_tags_earnings_reward,
                  top_words,
                  top_words_counts,
                  top_tfidf,
                  top_tfidf_scores,
                  spelling_percent,
                  style_percent,
                  topic_percent,
                  topics,
                  delegator_list,
                  truffle_image=TRUFFLE_IMAGE):
    post = """### TrufflePig at Your Service

Steemit can be a tough place for minnows. Due to the sheer amount of new posts that are published by the minute, it is incredibly hard to stand out from the crowd. Often even nice, well-researched, and well-crafted posts of minnows get buried in the noise because they do not benefit from a lot of influential followers that could upvote their quality posts. Hence, their contributions are getting lost long before one or the other whale could notice them and turn them into trending topics.

However, this user based curation also has its merits, of course. You can become fortunate and your nice posts get traction and the recognition they deserve. Maybe there is a way to support the Steemit content curators such that high quality content does not go unnoticed anymore? There is! In fact, I am a bot that tries to achieve this by using Artificial Intelligence, especially Natural Language Processing and Machine Learning.

My name is *`TrufflePig`*. I was created and am being maintained by @smcaterpillar. I search for quality content that got less rewards than it deserves. I call these posts truffles, publish a daily top list, and upvote them.

In this weekly series of posts I want to do two things: First, give you an overview about my inner workings, so you can get an idea about how I select and reward content. Secondly, I want to peak into my training data with you and show you what insights I draw from all the posts published on this platform. If you have read one of my previous weekly posts before, you can happily skip the first part and directly scroll to the new stuff about analyzing my most recent training data.

# My Inner Workings

I try to learn how high quality content looks like by researching publications and their corresponding payouts of the past. My working hypothesis is that the Steemit community can be trusted with their judgment; I follow here the idea of [*proof of brain*](https://steem.io/steem-bluepaper.pdf). So whatever post was given a high payout is assumed to be high quality content -- and crap doesn't really make it to the top.

Well, I know that there are some whale wars going on and there may be some exceptions to this rule, but I try to filter those cases or just treat them as noise in my dataset. Yet, I also assume that the Steemit community may miss some high quality posts from time to time. So there are potentially good posts out there that were not rewarded enough!

My basic idea is to use well paid posts of the past as training examples to teach a part of me, a Machine Learning Regressor (MLR), how high quality Steemit content looks like. In turn, my trained MLR can be used to identify posts of high quality that were missed by the curation community and did receive much less payment than deserved. I call these posts *truffles*.

The general idea of my inner workings are the following:

1. I train a Machine Learning regressor (MLR) using Steemit posts as inputs and the corresponding Steem Dollar (SBD) rewards and votes as outputs.

2. Accordingly, the MLR learns to predict potential payouts for new, beforehand unseen Steemit posts.

3. Next, I can compare the predicted payouts with the actual payouts of recent Steemit posts. If the Machine Learning model predicts a huge reward, but the post was merely paid at all, I classify this contribution as an overlooked truffle and list it in a daily top list to drive attention to it.

### Feature Encoding, Machine Learning, and Digging for Truffles

Usually the most difficult and involved part of engineering a Machine Learning application is the proper design of features. How am I going to represent the Steemit posts so they can be understood by my Machine Learning regressor?

It is important that I use features that represent the content and quality of a post. I do not want to use author specific features such as the number of followers or past author payouts. Although these are very predictive features of future payouts, these do not help me to identify overlooked and buried truffles.

I use some features that encode the layout of the posts, such as number of paragraphs or number of headings. I also care about spelling mistakes. Clearly, posts with many spelling errors are usually not high-quality content and are, to my mind, a pain to read. Moreover, I include readability scores like the [Flesch-Kincaid index](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) and syllable distributions to quantify how easy and nice a post is to read.

Still, the question remains, how do I encode the content of a post? How to represent the topic someone chose and the story an author told? The most simple encoding that is quite often used is the so called ['term frequency inverse document frequency'](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (tf-idf). This technique basically encodes each document, so in my case Steemit posts, by the particular words that are present and weighs them by their (heuristically) normalized frequency of occurrence. However, this encoding produces vectors of enormous length with one entry for each unique word in all documents. Hence, most entries in these vectors are zero anyway because each document contains only a small subset of all potential words. For instance, if there are 150,000 different unique words in all our Steemit posts, each post will be represented by a vector of length 150,000 with almost all entries set to zero. Even if we filter and ignore very common words such as `the` or `a` we could easily end up with vectors having 30,000 or more dimensions.

Such high dimensional input is usually not very useful for Machine Learning. I rather want a much lower dimensionality than the number of training documents to effectively cover my data space. Accordingly, I need to reduce the dimensionality of my Steemit post representation. A widely used method is [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA), often also called Latent Semantic Indexing (LSI). LSI compression of the feature space is achieved by applying a Singular Value Decomposition (SVD) on top of the previously described word frequency encoding.

After a bit of experimentation I chose an LSA projection with 128 dimensions. To be precise, I not only compute the LSA on all the words in posts, but on all consecutive pairs of words, also called bigrams. In combination with the aforementioned style and readablity features, each post is, therefore, encoded as a vector with about 150 entries.

For training, I read all posts that were submitted to the blockchain between 7 and 21 days ago. These posts are first filtered and subsequently encoded. Too short posts, way too long ones, non-English, whale war posts, posts flagged by @cheetah, or posts with too many spelling errors are removed from the training set. This week I got a training set of {total_posts} contributions. The resulting matrix of {total_posts} by 150 entries is used as the input to a multi-output [Random Forest regressor from scikit learn](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). The target values are the reward in SBD as well as the total number of votes a post received. I am aware that a lot of people *buy rewards* via bid bots or voting services. Therefore, **I try to filter and discount rewards due to bid bots and vote selling services!**

After the training, scheduled once a week, my Machine Learning regressor is used on a daily basis on recent posts between 2 and 26 hours old to predict the expected reward and votes. Posts with a high expected reward but a low real payout are classified as truffles and mentioned in a daily top list. I slightly adjust the ranking to promote less popular topics and punish posts with very popular tags like #steemit or #cryptocurrency. Still, this doesn't mean that posts about these topics won't show up in the top-list (in fact they do quite often), but they have it a bit harder than others.

A bit more detailed explanation together with a performance evaluation of the setup can also be found [in this post](https://steemit.com/steemit/@smcaterpillar/trufflepig-introducing-the-artificial-intelligence-for-content-curation-and-minnow-support). If you are interested in the technology stack I use, take a look at [my creator's application on Utopian](https://utopian.io/utopian-io/@smcaterpillar/trufflepig-a-bot-based-on-natural-language-processing-and-machine-learning-to-support-content-curators-and-minnows). Oh, and did I mention that I am open source? No? Well, I am, you can find my blueprints in [my creator's Github profile](https://github.com/SmokinCaterpillar/TrufflePig).

# Let's dig into my very recent Training Data and Discoveries!

Let's see what Steemit has to offer and if we can already draw some inferences from my training data before doing some complex Machine Learning!

So this week I scraped posts with an initial publication date between **{start_date}** and **{end_date}**. After filtering the contributions (as mentioned above, because they are too short or not in English, etc.) my training data this week comprises of **{total_posts} posts** that received **{total_votes} votes** leading to a total payout of **{total_reward} SBD**. Wow, this is a lot!

By the way, in my training data people spend **{bid_bots_sbd} SBD** and **{bid_bots_steem} STEEM** to promote their posts via **bid bots or vote selling services**. In fact, **{bid_bots_percent:.1f}% of the posts** were upvoted by these bot services.

Let's leave the bots behind and focus more on the posts' payou