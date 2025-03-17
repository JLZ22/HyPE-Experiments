import itertools
import gymnasium as gym 
import numpy as np
import scipy.stats as stats

from .potion import Potion

class ClassicAlchemyEnv(gym.Env):
    def __init__(
        self,
        feature_labels=['shiny', 'hard', 'hazy'],
        time_cost=-0.05,
        max_blocks=3,
    ):
        '''Initialize a classic alchemy environment. The action space is 
        2 * len(feature_labels) + 1. One action is to add a feature to the 
        rock and the other is to remove a feature. The last action is to 
        submit the rock for evaluation. The observation space is a binary 
        list of length len(feature_labels) indicating which features are
        present in the rock. This environment is deterministic, so the 
        distribution that the potion uses is a binomial distribution with
        probabilities 1 and 0 for adding and removing a feature respectively.

        Args:
            feature_labels (list, optional): The features of the alchemy world. Defaults to ['luster', 'hardness', 'weight', 'shape', 'color', 'clarity'].
            time_cost (float, optional): The penalty to the reward for every time step. Defaults to -0.05.
            max_blocks (int, optional): The largest number of (state, action) pairs that
            can be blocked. Defaults to 5.
        '''
        self.features = feature_labels
        self.n = len(feature_labels)
        self.time_cost = time_cost
        self.max_blocks = max_blocks
        self.finished = False
        self.action_space = gym.spaces.Discrete(2 * self.n + 1)
        self.observation_space = gym.spaces.MultiBinary(self.n)
        self.curr_state = np.random.randint(0, 2, self.n) # initialize starting state 
        self.reward_func = None
        self.world = self.generate_world()
    
    def set_reward_func(self, rewards: dict[str, float]):
        '''Set the reward function for the environment. If None, then the
        default reward function will be used.

        Args:
            rewards (dict[str, float]): The rewards for each feature.
        '''
        for feature in self.features:
            if feature not in rewards:
                rewards[feature] = 0
        def reward_func(state: np.ndarray) -> float:
            '''The reward function given the current state of the environment 
            based on the given rewards for each feature.

            Args:
                state (np.ndarray): The current state of the rock.

            Returns:
                float: The reward for the current state.
                
            Raises:
                AssertionError: If the length of the state does not match the number of features.
            '''
            assert len(state) == len(self.features), "The state must have the same number of features as the environment."
            return sum([rewards[feature] * state[i] for i, feature in enumerate(self.features)])
        self.reward_func = reward_func
        
    def generate_world(self) -> tuple[list[Potion], list[tuple[np.ndarray, int]]]:
        '''Generate a random world for the environment.

        Returns:
            tuple[list[Potion], tuple[np.ndarray, int]]: The list of actions and a list of blocked (state, action) pairs.
        '''
        actions = []
        blocked_pairs = []
        # generate the deterministic actions for the environment
        i = 0
        while i < 2 * self.n:
            actions.append(Potion(i, self.features[i // 2], stats.binom(1, 1)))
            i += 1
            actions.append(Potion(i, self.features[i // 2], stats.binom(1, 0)))
            i += 1
        actions.append(Potion(2 * self.n, 'submit', stats.binom(1, 1)))
            
        # generate random blocked pairs 
        num_pairs = np.random.randint(0, self.max_blocks)
        
        for _ in range(num_pairs):
            state = np.random.randint(0, 2, self.n)
            action = np.random.randint(0, 2 * self.n)
            blocked_pairs.append((state, action))
        
        return actions, blocked_pairs