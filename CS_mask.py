import numpy as np
from numpy.lib.stride_tricks import as_strided

def cartesian_mask(shape, acc, sample_n=10, centred=True):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - (Ncines, Nx, Ny, Ntime)
    shape: tuple - (Ncines, Ntime, Nx, Ny)
    acc: float - accleration factor, doesn't have to be integer 4, 8, etc..
    """

    def normal_pdf(length, sensitivity):
        return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)
    
    
    Ncines, Ntime, Nx, Ny = shape
    N = Ncines * Ntime

    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1. / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape((Ncines, Ntime, Nx, Ny))

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(2, 3))

    return mask