import gymnasium as gym
import torch 

from typing import Tuple, List
from src.utils import to_one_hot

def run_actions(
    model: torch.nn.Module,
    actions: List[int],
    curr_state: torch.Tensor,
    action_space: gym.spaces.Discrete,
    ) -> Tuple[List[torch.Tensor], int, int]:
    '''Run the given actions on the model and 
    return the final state sequence, total reward and
    number of non-terminal states.

    Args:
        model (torch.nn.Module): The model of the environment. 
        actions (List[int]): The sequence of actions.
        curr_state (torch.Tensor): The current state of the environment.
        action_space (gym.spaces.Discrete): The action space of the environment.

    Returns:
        Tuple[List[torch.Tensor], int, int]: state sequence, total reward, number of non-terminal states.
    '''
    state_seq = [curr_state]
    total_reward = 0
    num_non_term_states = 0
    finished = False
    for action in actions:
        if not finished:
            action_one_hot = to_one_hot(action, action_space.n)
            next_state, reward, term, _ = model.run(curr_state, action_one_hot)
            finished = term > 0.5 
            total_reward += reward.item()   
            curr_state = next_state.detach()
            if not finished:
                num_non_term_states += 1
        state_seq.append(curr_state)
    return state_seq, total_reward, num_non_term_states