import pandas as pd

POSTS = [
    {
            'title': 'Ladida',
            'reward': 42,
            'votes': 1,
            'active_votes': [{'voter': 'marc', 'percent': 42}],
            'created': pd.to_datetime('2018-01-01'),
            'tags': ['ff', 'kk'],
            'body': ('Mery had little l