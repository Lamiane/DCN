import theanets as t
import numpy as np
import notificator

T = 20  # number of time steps
K = 3  # number of values
BATCH_SIZE = 32  # this line should be out # why?


def generate():
    s, tk = np.random.randn(2, T, BATCH_SIZE, 1).astype('f')
    s[:K] = tk[-K:] = np.random.randn(K, BATCH_SIZE, 1)
    return [s, tk]


exp = t.Experiment(t.recurrent.Regressor, layers=(1, ('lstm', 10), 1), batch_size=BATCH_SIZE)
exp.train(generate, optimize="rmsprop")
exp.save("recurrent.p")

notificator.notify()