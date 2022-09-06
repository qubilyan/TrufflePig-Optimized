
import pandas as pd
import sqlite3
import json
import logging
import gc


logger = logging.getLogger(__name__)


JSON = 'JSON_'
SETJSON = 'SETJSON_'
TIMESTAMP = 'TIMESTAMP_'


def to_sqlite(frame, filename, tablename, index=True):
    """ Stores a data frame to sqlite

    Parameters
    ----------
    frame: DataFrame
    filename: str
    tablename: str
    index: bool

    """
    con = sqlite3.connect(filename)

    store_frame = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        first_element = frame[col].iloc[0]
        if isinstance(first_element, (list, dict, tuple)):
            store_frame[JSON+col] = frame[col].apply(json.dumps)
        elif isinstance(first_element, set):
            store_frame[SETJSON+col] = frame[col].apply(lambda x:
                                                        json.dumps(tuple(x)))
        elif isinstance(first_element, pd.Timestamp):
            store_frame[TIMESTAMP+col] = frame[col]
        else:
            store_frame[col] = frame[col]

    logger.info('Storing as SQLITE file {}'.format(filename))
    store_frame.to_sql(tablename, con, index=index)

    logger.info('Garbage Collecting')
    del store_frame
    gc.collect()

