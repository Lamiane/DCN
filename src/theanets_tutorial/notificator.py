import subprocess as s


# will stop playing after entering 'q' and hitting Enter
def notify():
    s.call(["mplayer", "-slave", "-quiet", "../media/crazy_mexican.mp3"])
