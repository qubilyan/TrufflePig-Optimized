import pandas as pd

import trufflepig.preprocessing as tppp
from trufflepig.testutils.random_data import create_n_random_posts
from trufflepig.testutils.raw_data import POSTS


def test_preprocessing():
    post_frame = pd.DataFrame(POSTS)
    filtered = tppp.preprocess(post_frame, ncores=1, min_en_prob=0.5,
                               max_errors_per_word=0.5,
                               min_max_num_words=(10, 99999))

    assert len(fi