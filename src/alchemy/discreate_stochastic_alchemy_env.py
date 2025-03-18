import numpy as np
import scipy.stats as stats

from .alchemy_world import AlchemyWorld
from .classic_alchemy_env import ClassicAlchemyEnv
from .potion import Potion

class DiscreteStochasticAlchemyEnv(ClassicAlchemyEnv):
    '''This is a subclass of ClassicAlchemyEnv that has 
    stochastic state transitions which are discrete meaning
    that features can either be added or removed. 
    '''
    
    def __init__(
        self,
        feature_labels = ['shiny', 'hard', 'hazy'],
        time_cost = -0.05,
        max_blocks = 3,
    ):
        '''Initialize a discrete stochastic alchemy environment. The 
        action space is 2 * len(feature_labels) + 1. This differs from
        the classic alchemy environment in that actions are now stochastic
        instead of deterministic (the Bernoulli distributions have probabilities
        that are in the range [0,1] instead of the set {0, 1}). Any given potion 
        is still uni-directional meaning that it can only either add or remove 
        a feature. The observation space is a binary list of length len(feature_labels) 
        indicating which features are present in the rock.

        Args:
            feature_labels (list, optional): _description_. Defaults to ['shiny', 'hard', 'hazy'].
            time_cost (float, optional): _description_. Defaults to -0.05.
            max_blocks (int, optional): _description_. Defaults to 3.
        '''
        # Can use super() because the action and observation spaces are the same
        # as the classic alchemy environment. This does not apply to bi-directional
        # and continuous alchemy environments because the action space in bi-directional
        # environments is n + 1 and the observation space in continuous environments is 
        # is populated with floats instead of binary values.
        super().__init__(feature_labels, time_cost, max_blocks)
        
    def generate_world(self) -> AlchemyWorld:
        '''Generate a random world for the environment using 
        random probabilities for each potion.

        Returns:
            AlchemyWorld: The world that stores actions, 
            transition dynamics, and blocked (state, action) pairs.
        '''
        probabilities = np.random.random(self.num_features)
        return self.generate_from_probabilities(probabilities)
        
    def generate_from_probabilities(
        self, 
        feature_probabilities: list[float]
    ) -> AlchemyWorld:
        '''Generate an alchemy world with self.action_space.n actions
        where the probability of 1 for a features[i] is given by 
        feature_probabilities[i * 2] and the probability of 0 for
        features[i] is given by 1 - feature_probabilities[i * 2]. 

        Args:
            feature_probabilities (list[float]): A list of probabilities of success for each feature. 
            This must be a list of length self.num_features.

        Returns:
            AlchemyWorld: The world that stores actions, 
            transition dynamics, and blocked (state, action) pairs.
    
        Raises:
            AssertionError: If the length of feature_probabilities is not equal to self.num_features.
        '''
        assert len(feature_probabilities) == self.num_features, 'You must provide a probability of success for each feature.'
        actions = []
        blocked_pairs = []
        # Add an action for each feature according to the given probabilities
        for i in range(self.num_features):
            p = feature_probabilities[i]
            actions.append(Potion(i, 1, stats.binom(1, p)))
            actions.append(Potion(i, 0, stats.binom(1, 1 - p)))
        actions.append(Potion(2 * self.num_features, 'submit', stats.binom(1, 1)))
            
        # Add random blocked (state, action) pairs
        num_pairs = np.random.randint(0, self.max_blocks)
        for _ in range(num_pairs):
            blocked_pairs.append(self._generate_random_state_action_pair())
            
        return AlchemyWorld(actions, blocked_pairs)