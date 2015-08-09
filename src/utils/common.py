# some useful utilities not specific for any module
import itertools
from copy import copy


# will stop playing after entering 'q' and hitting Enter
def notify():
    import subprocess as s
    s.call(["mplayer", "-slave", "-quiet", "../../media/crazy_mexican.mp3"])


# returns string with current time formatted as YEAR-MONTH-DAY HOURS:MINUTES:SECONDS
def get_timestamp():
    import time
    import datetime
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
    return st


def prime_factors(n):
    """Finds the prime factors of 'n'"""
    from math import sqrt

    pFact, limit, check, num = [], int(sqrt(n)) + 1, 2, n
    if n == 1:
        return [1]
    for check in range(2, limit):
        while num % check == 0:
            pFact.append(check)
            num /= check
    if num > 1:
        pFact.append(num)
    return pFact


def intersection(pfa, pfb):
    # c = [x for x in pfa if x in pfb]
    # c = list(set(x)) # usuwanie powtorzen
    c = []
    tmpb = copy(pfb)
    for i in pfa:
        if i in tmpb:
            c.append(i)
            for k in range(len(tmpb)):
                if tmpb[k] == i:
                    if k == 0:
                        tmpb = tmpb[1:]
                    elif k == len(tmpb):
                        tmpb = tmpb[:-1]
                    else:
                        ntmpb = tmpb[:k]
                        ntmpb.extend(tmpb[k + 1:])
                        tmpb = ntmpb
                    break
    return c


def combinations(intersection):
    for s in xrange(0, len(intersection) + 1):
        for comb in itertools.combinations(intersection, s):
            yield comb



