import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from solver_gridworld import PolicyIteration, ValueIteration, SARSA, QLearning

def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # constants and parameters
    theta=0.0005
    gamma=0.95
    eps=0.8
    alpha=0.2
    num_episodes=10000

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'mean-value-PI': [],
        'iter-PI': [0],
        'mean-value-VI': [],
        'iter-VI': [0],
        'returns-sarsa': [],
        'episode-sarsa': [0],
        'returns-qlearning': [],
        'episode-qlearning': [0],

    }

    # Policy Iteration
    PI_agent=PolicyIteration(theta,gamma,env)
    PI_values,PI_policy=PI_agent.learn_policy(log)
    print('Policy Iteration:',PI_values,PI_policy.policy)

    # Value Iteration
    VI_agent=ValueIteration(theta,gamma,env)
    VI_values,VI_policy=VI_agent.learn_policy(log)
    print('Value Iteration:',VI_values,VI_policy.policy)

    # SARSA
    sarsa_agent=SARSA(theta, gamma, alpha, eps, num_episodes, env)
    sarsa_values, sarsa_policy=sarsa_agent.learn_policy()
    print('SARSA:',sarsa_values,sarsa_policy.policy)

    # Q-Learning
    QLearning_agent=QLearning(theta, gamma, alpha, eps, num_episodes, env)
    QLearning_values, QLearning_policy=QLearning_agent.learn_policy()
    print('Q-Learning:',QLearning_values,QLearning_policy.policy)

    

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(4)
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    # plt.plot(log['t'], log['s'])
    plt.figure(1)
    plt.plot(log['iter-PI'], log['mean-value-PI'])
    plt.plot(log['iter-VI'], log['mean-value-VI'])
    plt.show()
    plt.figure(2)
    plt.plot(PI_policy.policy)
    plt.plot(VI_policy.policy)
    plt.plot(sarsa_policy.policy)
    plt.plot(QLearning_policy.policy)
    plt.legend(['PI', 'VI', 'SARSA', 'Q-Learning'])
    plt.show()
    # plt.plot(log['t'][:-1], log['a'])
    # plt.plot(log['t'][:-1], log['r'])
    # plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


if __name__ == '__main__':
    main()
