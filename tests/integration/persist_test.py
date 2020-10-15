import os
import pandas as pd

from trufflepig.testutils.pytest_fixtures import temp_dir
from trufflepig.testutils.random_data import create_n_random_posts
import trufflepig.preprocessing as tppp
import trufflepig.persist as tppe


def test_s