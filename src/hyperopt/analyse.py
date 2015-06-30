"""Generate data with test_parser.py, pipe the output to some_file (python test_parser.py > some_file).
Run: cat some_file | grep -E '(misclass|ITERATION)' > ITmis to generate ITmis file.
Now you can run this script on ITmis file.
"""

from re import compile
import matplotlib.pyplot as plt
from numpy import mean

with open("ITmis") as f:
    raw_data = f.read().split('\n')

results = []
idx = -1

# processing the file
digit_finder = compile(r'\d+.\d+')

for line in raw_data:
    # either create a list for a new experiment
    if 'ITERATION' in line:
        idx += 1
        results.append([])
    # or add a new misclass error to the experiment now being processed
    else:
        matched = digit_finder.search(line)
        if matched is not None:
            results[idx].append(float(matched.group()))

minimums = []
# for each experiment that has been run
for experiment in results:
    # if it has seen at least 3 epochs
    if len(experiment) > 2:
        # plot the misclass error w.r.t number of epochs seem
        plt.plot(range(1, len(experiment)+1), experiment, 'ro')
        plt.show()
        # find minimal misclass error for this experiment
        minimums.append(min(experiment))

print "minimums:", minimums
print "mean", mean(minimums)