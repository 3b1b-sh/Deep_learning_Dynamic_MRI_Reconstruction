import itertools

from .utils import imgshow, imsshow
from .utils import compute_num_params as compute_params
from .dataset import FastmriKnee, DatasetReconMRI
from .dataset import arbitrary_dataset_split as split_dataset
from .utils import complex2pseudo, pseudo2real, pseudo2complex, kspace2image, image2kspace

from .models import DataConsistencyLayer

from .solver import Lab1Solver as Solver


def fetch_batch_sample(loader, idx):
    batch = next(itertools.islice(loader, idx, None))
    return batch
