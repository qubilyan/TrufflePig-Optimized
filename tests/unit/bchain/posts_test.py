import pandas as pd

import trufflepig.bchain.posts as tbpo
import trufflepig.filters.textfilters as tptf
import trufflepig.preprocessing as tppp
from trufflepig.testutils import random_data


def test_comment():
    post = random_data.create_n_random_posts(1)[0]

    result = tbpo.truffle_comment(reward=post['reward'],
                                  votes=post['votes'],
                                  rank=1,
                                  topN_link='www.example.com',