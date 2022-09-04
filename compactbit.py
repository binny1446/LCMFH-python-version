import numpy as np


def compactbit(b=None,*args,**kwargs):
    varargin = compactbit.varargin
    nargin = compactbit.nargin

    nSamples, nbits=np.shape(b)

    nwords=np.ceil(nbits / 8)

    cb=np.zeros(([nSamples,nwords]), dtype = np.uint16 )

    for j in np.arange(1,nbits).reshape(-1):
        w=np.ceil(j / 8)

        cb[np.arange(),w]=bitset(cb(np.arange(),w),(j-1)%8 + 1,b(np.arange(),j))

    