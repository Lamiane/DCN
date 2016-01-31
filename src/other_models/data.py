import pickle as pkl
from numpy import ravel, vstack
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
import sys
sys.path.append('..')
from stefan.sift2d import *


# takes dat ifle with sift2d objects and saves them as sparse matrices
# important note: samples are raveled to work well with SVNs and such models
# data is saved under same name but with 'npy' extension
def dat2npy(filename):
    f = open(filename)
    dl = pkl.load(f)
    # raveling for svm and other such models
    dna = list([ravel(element.get_numpy_array()) for element in dl])[0:2]
    # type(dna)
    #  <type 'list'>

    data = csr_matrix(vstack(dna))
    print data.shape
    # (n_samples, sample_length)
    # actives: (2728, 19782)
    # inactives: (1985, 19782)
    # middle: (1248, 19782)

    ff = open(filename+'.npy', 'w')
    # saving more than one ndarray with np.savez 'cause we're dealing with csr_matrix
    np.savez(ff, data=data.data, indices=data.indices, indptr=data.indptr, shape=data.shape)
    ff.close()


# load npy and retrieve the csr_matrix that's inside
# the matrix that is returned is dense
def load(filename):
    loaded = np.load(filename)
    data = csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape']).todense()
    data = np.array(data)
    return data


def load_data(active_path, nonactive_path):
    actives = load(active_path)
    nonactives = load(nonactive_path)

    X, y = glue_and_generate_labels(actives, nonactives)

    return X, y


def glue_and_generate_labels(act, nact):
    act_y = np.ones(act.shape[0])
    nact_y = -np.ones(nact.shape[0])

    tr_X = np.vstack((act, nact))
    tr_y = np.hstack((act_y, nact_y))

    seed = 666
    X = shuffle(tr_X, random_state=seed)
    y = shuffle(tr_y, random_state=seed)

    return X, y


def dat2npy_1d(filename):
    with open(filename) as f:
        X = f.read().splitlines()

    X = [line.split(':') for line in X]
    X = np.array(X, dtype='O')[:, 3]
    X = np.array([list(el) for el in X], dtype='int')

    data = csr_matrix(X)
    print data.shape
    # (n_samples, sample_length)
    # actives: (2728, 19782)
    # inactives: (1985, 19782)
    # middle: (1248, 19782)

    with open(filename+'.npy', 'w') as ff:
    # saving more than one ndarray with np.savez 'cause we're dealing with csr_matrix
        np.savez(ff, data=data.data, indices=data.indices, indptr=data.indptr, shape=data.shape)
    return X, y