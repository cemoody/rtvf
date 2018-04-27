import sys
import scipy
import numpy as np
from model import fit
from sklearn.externals import joblib

mem = joblib.Memory('mem')


def load_image(fn, xshift=0, yshift=0, downsample=8, transpose=False):
    x = scipy.misc.imread(fn).astype('float32') / 255.
    m = int(min(x.shape[:2]) / 2)
    xc, yc = [int(s / 2) for s in x.shape[:2]]
    xc += xshift
    yc += yshift
    y = x[xc - m: xc + m, yc - m: yc + m, :]
    if downsample:
        y = y[::downsample, ::downsample, :]
    if transpose:
        y = y.transpose((2, 0, 1))
    assert y.shape[0] == y.shape[1]
    # grey = y.mean(axis=0)
    # flatten = np.ravel(grey)
    return y


@mem.cache
def load_all(fns):
    X = np.array([load_image(fn, yshift=-150) for fn in fns])
    return X


if __name__ == '__main__':
    fns = sys.argv[1:]
    fns = sorted(fns)
    X = load_all(fns)
    fit(X)
