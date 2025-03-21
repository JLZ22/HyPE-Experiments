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

def sepfunc_l2across(
    models: List[torch.nn.Module],
    actions: List[int],
    curr_state: torch.Tensor,
    action_space: gym.spaces.Discrete,
    avg_reward_weight: float = 0.0,
) -> float:
    '''Run the sequence of actions on every model 
    starting at curr_state and return the statewise
    L2-distance between the models.

    Args:
        models (List[torch.nn.Module]): The models to compare.
        actions (List[int]): The sequence of actions to run.
        curr_state (torch.Tensor): The current state of the environment.
        action_space (gym.spaces.Discrete): The action space of the environment.
        avg_reward_weight (float): The weight to give to the average reward.

    Returns:
        float: The statewise distance between all the models 
        over the given actions.
    '''
    dist = 0.0 
    state_sequences = [] 
    rewards = []
    for model in models:
        state_seq, reward, _ = run_actions(model, actions, curr_state, action_space)
        rewards.append(reward)
        state_sequences.append(state_seq)
    
    num_models = len(models)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            for seq1, seq2 in zip(state_sequences[i], state_sequences[j]):
                dist += torch.norm(seq1 - seq2).item()
    return dist + avg_reward_weight * sum(rewards) / len(rewards)

def sepfunc_central_deviation(
    models: List[torch.nn.Module],
    actions: List[int],
    curr_state: torch.Tensor,
    action_space: gym.spaces.Discrete,
    avg_reward_weight: float = 0.0,
) -> float:
    '''Run the sequence of actions on every model 
    starting at curr_state and return the central
    deviation between the models.

    Args:
        models (List[torch.nn.Module]): The models to compare.
        actions (List[int]): The sequence of actions to run.
        curr_state (torch.Tensor): The current state of the environment.
        action_space (gym.spaces.Discrete): The action space of the environment.
        avg_reward_weight (float): The weight to give to the average reward.

    Returns:
        float: The separation defined by the central 
        deviation function.
    '''
    state_sequences = [] 
    rewards = []
    for model in models:
        state_seq, reward, _ = run_actions(model, actions, curr_state, action_space)
        rewards.append(reward)
        state_sequences.append(state_seq)
    num_timesteps = len(actions) + 1
    total_separation = 0.0
    for t in range(num_timesteps):
        reference_state = sum(state_seq[t] for state_seq in state_sequences) / len(models)
        for state_seq in state_sequences:
            total_separation += torch.norm(state_seq[t] - reference_state).item()
    return total_separation + avg_reward_weight * sum(rewards) / len(rewards)