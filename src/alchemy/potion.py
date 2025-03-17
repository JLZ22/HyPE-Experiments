import numpy as np
import scipy.stats as stats
from typing import Union

class Potion():
    '''
    Potions represent the actions that the alchemist can take. Each potion
    can affect the state of a rock in a way that is determined by the 
    distribution of the potion. It can increase or decrease a certain 
    feature and only that feature. 
    '''
    
    def __init__(
        self,
        feature_idx: int,
        feature_name: str,
        distribution: Union[stats.rv_discrete, stats.rv_continuous],
        check_distr: bool = True,
    ):
        '''Initialize a potion that affects a certain feature. The feature 
        of a rock can be increased or decreased by the potion over the range 
        [0, 1]. The distribution of the potion determines how using it will 
        affect the feature. If you want discrete deterministic state transitions, use 
        a binomial distribution with n=1 and p=1 for adding the feature
        and p=0 for removing the feature.

        Args:
            feature_id (int): The id of the feature to affect.
            feature_name (str): The name of the feature to affect.
            distribution (stats.rv_discrete | stats.rv_continuous): The distribution of the potion. You may use any 
            statistical distribution from scipy.stats as long as the sampled values are within the range [0, 1].
            check_distr (bool, optional): Whether to check if the distribution is within the range [0, 1]. Defaults to True.
        
        Raises:
            ValueError: If the distribution is not within the range [0, 1].
        '''
        self.distribution = distribution
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        
        # check if the distribution is within the range [0, 1]
        if check_distr:
            samples = self.distribution.rvs(1000)
            if np.any(samples < 0) or np.any(samples > 1):
                raise ValueError('The distribution of the potion must be within the range [0, 1].')
            
    def use_on(self, state: np.ndarray) -> np.ndarray:
        '''Use this potion on the given state and 
        return the state of the rock after applying 
        the potion as a copy of the original state.

        Returns:
            np.ndarray: The new state of the rock.
        '''
        new_state = state.copy() 
        new_state[self.feature_idx] = self.distribution.rvs()
        return new_state