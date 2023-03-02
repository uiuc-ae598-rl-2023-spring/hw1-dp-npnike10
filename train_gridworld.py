import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from solver_gridworld import PolicyIteration, ValueIteration, SARSA

def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # constants and parameters
    theta=0.0005
    gamma=0.95
    eps=0.6
    alpha=0.3
    num_episodes=10000

    # Initialize simulation
    s = env.reset()

    # Policy Iteration
    PI_agent=PolicyIteration(theta,gamma,env)
    PI_values,PI_policy=PI_agent.learn_policy()
    print('Policy Iteration:',PI_values,PI_policy.policy)

    # Value Iteration
    VI_agent=ValueIteration(theta,gamma,env)
    VI_values,VI_policy=VI_agent.learn_policy()
    print('Value Iteration:',VI_values,VI_policy.policy)

    # SARSA
    sarsa_agent=SARSA(theta, gamma, alpha, eps, num_episodes, env)
    sarsa_policy=sarsa_agent.learn_policy()
    print('SARSA:',sarsa_policy.policy)

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

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
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


if __name__ == '__main__':
    main()
