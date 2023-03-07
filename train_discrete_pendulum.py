import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
from solver_gridworld import SARSA, QLearning

def test_x_to_s(env):
    theta = np.linspace(-np.pi * (1 - (1 / env.n_theta)), np.pi * (1 - (1 / env.n_theta)), env.n_theta)
    thetadot = np.linspace(-env.max_thetadot * (1 - (1 / env.n_thetadot)), env.max_thetadot * (1 - (1 / env.n_thetadot)), env.n_thetadot)
    for s in range(env.num_states):
        i = s // env.n_thetadot
        j = s % env.n_thetadot
        s1 = env._x_to_s([theta[i], thetadot[j]])
        if s1 != s:
            raise Exception(f'test_x_to_s: error in state representation: {s} and {s1} should be the same')
    print('test_x_to_s: passed')


def main():
    # Create environment
    #
    #   By default, both the state space (theta, thetadot) and the action space
    #   (tau) are discretized with 31 grid points in each dimension, for a total
    #   of 31 x 31 states and 31 actions.
    #
    #   You can change the number of grid points as follows (for example):
    #
    #       env = discrete_pendulum.Pendulum(n_theta=11, n_thetadot=51, n_tau=21)
    #
    #   Note that there will only be a grid point at "0" along a given dimension
    #   if the number of grid points in that dimension is odd.
    #
    #   How does performance vary with the number of grid points? What about
    #   computation time?
    env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21,n_tau=31) #15,21

    def wrap_pi(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    # constants and parameters
    theta=0.0005
    gamma=0.95
    eps=0.6
    alpha=0.1
    num_episodes=3000

    # Apply unit test to check state representation
    test_x_to_s(env)

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't-sarsa': [0],
        's-sarsa': [s],
        'a-sarsa': [],
        'r-sarsa': [],
        't-qlearning': [0],
        's-qlearning': [s],
        'a-qlearning': [],
        'r-qlearning': [],
        'returns-sarsa': [],
        'returns-qlearning': [],
        'theta-qlearning':[],
        'thetadot-qlearning':[],
        'theta-sarsa':[],
        'thetadot-sarsa':[]
    }

    # SARSA
    sarsa_agent=SARSA(theta, gamma, alpha, eps, num_episodes, env)
    sarsa_values, sarsa_policy=sarsa_agent.learn_policy(log)
    print('SARSA:',sarsa_values,sarsa_policy.policy)

    # Q-Learning
    QLearning_agent=QLearning(theta, gamma, alpha, eps, num_episodes, env)
    QLearning_values, QLearning_policy=QLearning_agent.learn_policy(log)
    print('Q-Learning:',QLearning_values,QLearning_policy.policy)

    # Simulate until episode is done - SARSA
    done = False
    env.reset()
    s=7
    log['s-sarsa'][0]=s
    while not done:
        a = sarsa_policy.get_action(s)
        (next_s, r, done) = env.step(a)
        log['t-sarsa'].append(log['t-sarsa'][-1] + 1)
        log['s-sarsa'].append(s)
        log['a-sarsa'].append(a)
        log['r-sarsa'].append(r)
        log['theta-sarsa'].append(wrap_pi(env.x[0]))
        log['thetadot-sarsa'].append(env.x[1])
        s=next_s
    
    # Simulate until episode is done - Q-Learning
    done = False
    env.reset()
    s=7
    log['s-qlearning'][0]=s
    log['theta-qlearning'].append(env.x[0])
    log['thetadot-qlearning'].append(env.x[1])
    while not done:
        a = QLearning_policy.get_action(s)
        (next_s, r, done) = env.step(a)
        log['t-qlearning'].append(log['t-qlearning'][-1] + 1)
        log['s-qlearning'].append(s)
        log['a-qlearning'].append(a)
        log['r-qlearning'].append(r)
        log['theta-qlearning'].append(wrap_pi(env.x[0]))
        log['thetadot-qlearning'].append(env.x[1])
        s=next_s

    # Plot data and save to png file
    fig4, ax = plt.subplots(2, 1, figsize=(10, 10))
    #ax[0].plot(log['t-qlearning'], log['s-qlearning'])
    ax[0].plot(log['t-qlearning'][:-1], log['a-qlearning'])
    ax[0].plot(log['t-qlearning'][:-1], log['r-qlearning'])
    ax[0].legend(['a', 'r'])
    ax[1].plot(log['t-qlearning'], log['theta-qlearning'])
    ax[1].plot(log['t-qlearning'], log['thetadot-qlearning'])
    ax[1].legend(['theta', 'thetadot'])
    plt.savefig('figures/pendulum/test_discrete_pendulum.png')
    plt.show()

    plt.figure(1)
    plt.plot(sarsa_policy.policy)
    plt.plot(QLearning_policy.policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title('Policy plot')
    plt.legend(['SARSA', 'Q-Learning'])
    plt.savefig('figures/pendulum/policies.png')
    plt.show()

    plt.figure(2)
    plt.plot(range(1,num_episodes+1),log['returns-sarsa'])      #TODO check if plotting with decayed eps would be useful
    plt.plot(range(1,num_episodes+1),log['returns-qlearning'])
    plt.xlabel('Number of Episodes')
    plt.ylabel('Return')
    plt.title('Learning Curve for Model-Free methods')
    plt.legend(['SARSA', 'Q-Learning'])
    plt.savefig('figures/pendulum/returns.png')
    plt.show()

    plt.figure(3)
    plt.plot(sarsa_values)                                      #TODO see if different value of eps,alpha makes sarsa curve coincide
    plt.plot(QLearning_values)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.title('Value Function')
    plt.legend(['SARSA', 'Q-Learning'])
    plt.savefig('figures/pendulum/values.png')
    plt.show()


if __name__ == '__main__':
    main()
