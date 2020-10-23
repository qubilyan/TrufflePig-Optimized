from trufflepig.testutils.pytest_fixtures import steem
import trufflepig.pigonduty as tppd
from tests.integration.model_test import MockPipeline
from trufflepig import config
from trufflepig.bchain.poster import Poster


def test_call_a_pig(steem):
    current_datetime = '2018-03-03-18:21:30'

    pipeline = MockPipel