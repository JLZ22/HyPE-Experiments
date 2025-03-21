import numpy as np 
import torch 

from typing import List  

def to_numpy(samples: List[torch.Tensor]) -> np.ndarray:
    arr = [sample.clone().detach().cpu() for sample in samples]
    return torch.cat(arr).numpy()

def multi_normal_kl_divergence(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
) -> float:
    '''Calculate the KLD of two multivariate normal distributions
    because we have a large sample size, CLT dictates that the samples
    will be approximately normally distributed.
    
    https://statproofbook.github.io/P/mvn-kl.html

    Args:
        samples_p (np.ndarray): Samples from the first distribution. This should be a 
            matrix where each row is a feature and each column is a sample.
        samples_q (np.ndarray): Samples from the second distribution. This should be a
            matrix where each row is a feature and each column is a sample.

    Returns:
        float: The KL divergence between the two distributions.
    '''
    num_features = samples_p.shape[0]
    mu_p = np.mean(samples_p, axis=1)
    mu_q = np.mean(samples_q, axis=1)
    cov_p = np.cov(samples_p)
    cov_q = np.cov(samples_q)
    
    term_1 = np.dot((mu_q - mu_p).T, np.dot(np.linalg.inv(cov_q), (mu_q - mu_p)))
    term_2 = np.trace(np.dot(np.linalg.inv(cov_q), cov_p))
    term_3 = - np.log(np.abs(len(cov_q) / len(cov_p))) - num_features
    
    return max(0.5 * (term_1 + term_2 + term_3), 0.0)

def approx_kld(
    samples_p: List[torch.Tensor], 
    samples_q: List[torch.Tensor],
) -> float:
    '''Calculate the Kullback-Leibler Divergence between the approximate
    distributions of the two sample sets. KLD is not symmetric, so the order
    of the sample sets matters. A large KLD means q is bad at approximating p
    while a small KLD means q is good at approximating p. Sample size should be 
    1000 - 10000 for good results.

    Args:
        samples_p (List[torch.Tensor]): The samples from the first distribution.
        samples_q (List[torch.Tensor]): The samples from the second distribution.

    Returns:
        float: The KL divergence between the two distributions.
    '''
    sp = to_numpy(samples_p).T
    sq = to_numpy(samples_q).T
    
    return multi_normal_kl_divergence(sp, sq)
    
def approx_jsd(
    samples_p: List[torch.Tensor], 
    samples_q: List[torch.Tensor],
) -> float:
    '''Calculate the Kullback-Leibler Divergence between the approximate
    distributions of the two sample sets. JSD is symmetric, so the order
    of the sample sets does not matter. A large JSD means the distributions
    are different while a small JSD means the distributions are similar.
    
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    Args:
        samples_p (List[torch.Tensor]): The samples from the first distribution.
        samples_q (List[torch.Tensor]): The samples from the second distribution.

    Returns:
        float: The KL divergence between the two distributions.
    '''    
    sp = to_numpy(samples_p).T
    sq = to_numpy(samples_q).T
    m = 0.5 * (sp + sq)
    
    return max(0.5 * approx_kld(samples_p, m) + 0.5 * approx_kld(samples_q, m), 0.0)