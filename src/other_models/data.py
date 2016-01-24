import pickle as pkl
import numpy as np
from numpy import ravel, vstack
from scipy.sparse import csr_matrix
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
    dna = list([ravel(element.get_numpy_array()) for element in dl])
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
