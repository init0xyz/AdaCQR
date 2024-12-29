import datetime
import logging
import sys

from pytz import timezone

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

def get_logger(name=None):
    if not name:
        name = 'main'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    def customTime(*args):
        utc_dt = datetime.datetime.now()
        my_tz = timezone("Europe/London")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    logging.Formatter.converter = customTime
    return logger
