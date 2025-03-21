import numpy as np 
import pytest 
import scipy.stats as stats
import torch 

from src.sep_funcs.distance_methods import approx_kld, approx_jsd

NUM_FEATURES = 6
NUM_SAMPLES = 1000
NUM_TRIALS = 50
NUM_DISTR = 3

def generate_and_reshape_samples(distr):
    samples = np.array([distr[i].rvs(NUM_SAMPLES) for i in range(NUM_FEATURES)]).T
    assert samples.shape == (NUM_SAMPLES, NUM_FEATURES)
    samples = torch.Tensor(samples).unsqueeze(1)
    assert samples.shape == (NUM_SAMPLES, 1, NUM_FEATURES)
    return samples

@pytest.mark.timeout(10)
def test_kld():
    '''Initialize a list of distributions and generate samples from them.
    Then, calculate the KLD between the first distribution and the rest
    including itself. Check that the KLD is non-negative for all and that 
    the KLD increases as the distributions become more different.
    '''
    distrs = []
    for i in range(NUM_DISTR):
        distrs.append([stats.beta(1 + i, 2)] * NUM_FEATURES)
    
    for _ in range(NUM_TRIALS):
        # get the samples from the distributions
        samples = []
        for d in distrs:
            samples.append(generate_and_reshape_samples(d))
        
        # calculate the KLDs between the first distribution and the rest
        klds = []
        for sample in samples:
            kld = approx_kld(samples[0], sample)
            assert isinstance(kld, float), "KLD should return a float."
            assert kld >= 0, "KLD should be non-negative."
            klds.append(kld)
        
        # check that the KLD of a distribution to itself is approximately zero
        assert klds[0] < 10e-6, "KLD of a distribution to itself should be approximately zero."
        # check that the KLDs are increasing
        prev_kld = klds[0]
        for kld in klds[1:]:
            abs_diff = np.abs(kld - prev_kld)
            assert kld >= prev_kld or abs_diff <= 10e-6, "Approximate KLD for same distribution should be approxmately less than or equal to the KLD for different distributions."
            prev_kld = kld
    
def test_jsd():
    ...