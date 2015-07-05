# some useful not specific utilities


# returns string with current time formatted as YEAR-MONTH-DAY HOURS:MINUTES:SECONDS
def get_timestamp():
    import time
    import datetime
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return st

