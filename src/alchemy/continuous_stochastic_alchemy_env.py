import gymnasium as gym
import numpy as np
import scipy.stats as stats

from typing import List
from .alchemy_world import AlchemyWorld
from .classic_alchemy_env import ClassicAlchemyEnv
from .potion import Potion

class ContinuousStochasticAlchemyEnv(ClassicAlchemyEnv):
    '''This is a subclass of ClassicAlchemyEnv that has
    stochastic state transitions which are continuous meaning
    that features can be increased or decreased by any amount
    between 0 and 1. 
    '''
    
    def __init__(
        self,
        feature_labels = ['luster', 'hardness', 'clarity'],
        time_cost = -0.05,
        max_blocks = 3,
        block_tolerance = 0.1,
    ):
        '''Initialize a continuous stochastic alchemy environment. The 
        action space is len(feature_labels) + 1. This differs from classic
        alchemy because the actions are stochastic and bi-directional. All
        features can be increased or decreased by any amount between 0 and 1
        by exactly one potion. The observation space is a list of floats of
        length len(feature_labels) indicating the values of each feature in
        the rock. Blocks are implemented slightly differently. Since the 
        observation space is continuous, the probability of a blocked state
        matching the current state is functionally zero. Therefore, a (state, action)
        will be blocked if the current state is within block_tolerance of the
        blocked state. 

        Args:
            feature_labels (list, optional): The names of the features.. Defaults to ['luster', 'hardness', 'clarity'].
            time_cost (float, optional): The cost of performing a non-ending action.. Defaults to -0.05.
            max_blocks (int, optional): The number of blocked state-action pairs. Defaults to 3.
            block_tolerance (float, optional): The tolerance for blocked state-action pairs. Defaults to 0.1.
        '''
        # since the action and observation spaces are different, we cannot use super()
        self.features = feature_labels
        self.num_features = len(feature_labels)
        self.time_cost = time_cost
        self.max_blocks = max_blocks
        self.block_tolerance = block_tolerance
        self.finished = False
        self.action_space = gym.spaces.Discrete(self.num_features + 1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=float)
        self.stale_actions = set() # set of actions that have been already taken in this episode
        
        self.curr_state = None 
        self.reward_func = None
        self.world = self.generate_world()
        self.reset()
        
    def is_blocked(self, state: np.ndarray, action: int) -> bool:
        '''Check if the given state-action pair is blocked.

        Args:
            state (np.ndarray): The current state of the rock.
            action (int): The action to take.

        Returns:
            bool: True if the state-action pair is blocked, False otherwise.
        '''
        for blocked_state, blocked_action in self.world.blocked_pairs:
            if (
                np.all(np.abs(state - blocked_state) < self.block_tolerance) 
                and 
                action == blocked_action
            ):
                return True
        return False
    
    def generate_world(self) -> AlchemyWorld:
        '''Generate a random world for the environment using 
        random probabilities for each potion.

        Returns:
            AlchemyWorld: The world that stores actions, 
            transition dynamics, and blocked (state, action) pairs.
        '''
        stds = np.random.uniform(0.1, 0.91, size=self.num_features)
        locs = np.random.uniform(0.1, 0.91, size=self.num_features)
        a = (0 - locs) / stds
        b = (1 - locs) / stds
        distributions = [
            stats.truncnorm(a[i], b[i], loc=locs[i], scale=stds[i]) for i in range(self.num_features)
        ]
        return self.generate_world_from_distributions(distributions)

    def generate_world_from_distributions(
        self, 
        distributions: List[stats.rv_continuous]
    ) -> AlchemyWorld:
        '''Generate an alchemy world with self.action_space.n actions
        where the distribution of the potion for each feature is given by
        distributions. Distributions must be a list of length self.num_features.

        Args:
            distributions (List[stats.rv_continuous]): The distributions of the potions for each feature.

        Returns:
            AlchemyWorld: The world that stores actions,
            transition dynamics, and blocked (state, action) pairs.
            
        Raises:
            AssertionError: You must provide a distribution for each feature.
        '''
        assert len(distributions) == self.num_features, 'You must provide a distribution for each feature.'
        actions = []
        blocked_pairs = []
        # Add an action for each feature according to the given distributions
        for i in range(self.num_features):
            actions.append(Potion(i, self.features[i], distributions[i]))
        actions.append(Potion(self.num_features, 'submit', stats.binom(1, 1)))
        
        # Add random blocked (state, action) pairs
        num_pairs = np.random.randint(0, self.max_blocks)
        for _ in range(num_pairs):
            blocked_pairs.append(self._generate_random_state_action_pair())
            
        return AlchemyWorld(actions, blocked_pairs)