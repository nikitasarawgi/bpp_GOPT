import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path) 

import random 

import gymnasium as gym
import torch
from tianshou.utils.net.common import ActorCritic

from ts_train import build_net
import arguments
from tools import *
from mycollector import PackCollector
from masked_ppo import MaskedPPOPolicy


def test(args):

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")
        
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # environment
    test_env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type=args.env.box_type,
        item_set=args.env.box_size_set, 
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=args.render,
        use_test_indices=True,
        save_img_path=args.test_log_file,
    )

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    
    # RL agent 
    dist = CategoricalMasked

    policy = MaskedPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.train.gamma,
        eps_clip=args.train.clip_param,
        advantage_normalization=False,
        vf_coef=args.loss.value,
        ent_coef=args.loss.entropy,
        gae_lambda=args.train.gae_lambda,
        action_space=test_env.action_space,
    )
    
    policy.eval()
    try:
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        # print(policy)
    except FileNotFoundError:
        print("No model found")
        exit()

    test_collector = PackCollector(policy, test_env)

    # Evaluation
    result = test_collector.collect(n_episode=args.test_episode, render=args.render)
    for i in range(args.test_episode):
        print(f"episode {i+1}\t => \tratio: {result['ratios'][i]:.4f} \t| total: {result['nums'][i]}")
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(result['ratio']))
    print('average put item number: %.4f'%(result['num']))
    print("standard variance: %.4f"%(result['ratio_std']))
    # Write these to the log file
    filepath = os.path.join(args.test_log_file, "test_summary.txt")
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        for i in range(args.test_episode):
            f.write(f"episode {i+1} => ratio: {result['ratios'][i]:.4f}, total: {result['nums'][i]}\n")
        f.write(f"average space utilization: {result['ratio']:.4f}\n")
        f.write(f"average put item number: {result['num']:.4f}\n")
        f.write(f"standard variance: {result['ratio_std']:.4f}\n")
        


if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  
    if args.render:
        args.test_episode = 1  # for visualization

    args.seed = 5
    print(f"dimension: {args.env.container_size}")
    test(args)
