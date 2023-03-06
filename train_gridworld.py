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
    eps=0.7
    alpha=0.1
    num_episodes=1500

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't-PI': [0],
        's-PI': [s],
        'a-PI': [],
        'r-PI': [],
        't-VI': [0],
        's-VI': [s],
        'a-VI': [],
        'r-VI': [],
        't-sarsa': [0],
        's-sarsa': [s],
        'a-sarsa': [],
        'r-sarsa': [],
        't-qlearning': [0],
        's-qlearning': [s],
        'a-qlearning': [],
        'r-qlearning': [],
        'mean-value-PI': [],
        'iter-PI': [0],
        'mean-value-VI': [],
        'iter-VI': [0],
        'returns-sarsa': [],
        'returns-qlearning': [],
        'returns-sarsa2': [],
        'returns-qlearning2': [],
        'returns-sarsa3': [],
        'returns-qlearning3': [],
        'returns-sarsa4': [],
        'returns-qlearning4': [],
        'returns-sarsa5': [],
        'returns-qlearning5': [],

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
    sarsa_values, sarsa_policy=sarsa_agent.learn_policy(log)
    print('SARSA:',sarsa_values,sarsa_policy.policy)

    # Q-Learning
    QLearning_agent=QLearning(theta, gamma, alpha, eps, num_episodes, env)
    QLearning_values, QLearning_policy=QLearning_agent.learn_policy(log)
    print('Q-Learning:',QLearning_values,QLearning_policy.policy)

    

    # Simulate until episode is done - PI
    done = False
    env.reset()
    s=7                                             # choosing same (arbitrary) starting state for all algorithms
    log['s-PI'][0]=s
    while not done:
        a = PI_policy.get_action(s)
        (next_s, r, done) = env.step(a)
        log['t-PI'].append(log['t-PI'][-1] + 1)
        log['s-PI'].append(s)
        log['a-PI'].append(a)
        log['r-PI'].append(r)
        s=next_s

    # Simulate until episode is done - VI
    done = False
    env.reset()
    s=7
    log['s-VI'][0]=s
    while not done:
        a = VI_policy.get_action(s)
        (next_s, r, done) = env.step(a)
        log['t-VI'].append(log['t-VI'][-1] + 1)
        log['s-VI'].append(s)
        log['a-VI'].append(a)
        log['r-VI'].append(r)
        s=next_s
    

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
        s=next_s
    
    # Simulate until episode is done - Q-Learning
    done = False
    env.reset()
    s=7
    log['s-qlearning'][0]=s
    while not done:
        a = QLearning_policy.get_action(s)
        (next_s, r, done) = env.step(a)
        log['t-qlearning'].append(log['t-qlearning'][-1] + 1)
        log['s-qlearning'].append(s)
        log['a-qlearning'].append(a)
        log['r-qlearning'].append(r)
        s=next_s

    # Plot data and save to png file
    plt.figure(1)
    plt.plot(log['iter-PI'], log['mean-value-PI'])
    plt.plot(log['iter-VI'], log['mean-value-VI'])
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean of Value Function')
    plt.title('Learning Curve for Model-Based methods')
    plt.savefig('figures/gridworld/mean-values.png')
    plt.show()


    plt.figure(2)
    plt.plot(PI_policy.policy)
    plt.plot(VI_policy.policy)
    plt.plot(sarsa_policy.policy)
    plt.plot(QLearning_policy.policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title('Policy plot')
    plt.legend(['PI', 'VI', 'SARSA', 'Q-Learning'])
    plt.savefig('figures/gridworld/policies.png')
    plt.show()

    plt.figure(3)
    plt.plot(range(1,num_episodes+1),log['returns-sarsa'])      #TODO check if plotting with decayed eps would be useful
    plt.plot(range(1,num_episodes+1),log['returns-qlearning'])
    plt.xlabel('Number of Episodes')
    plt.ylabel('Return')
    plt.title('Learning Curve for Model-Free methods')
    plt.legend(['SARSA', 'Q-Learning'])
    plt.savefig('figures/gridworld/returns.png')
    plt.show()

    plt.figure(4)
    plt.plot(PI_values)
    plt.plot(VI_values)
    plt.plot(sarsa_values)                                      #TODO see if different value of eps,alpha makes sarsa curve coincide
    plt.plot(QLearning_values)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.title('Value Function')
    plt.legend(['PI', 'VI', 'SARSA', 'Q-Learning'])
    plt.savefig('figures/gridworld/values.png')
    plt.show()

    plt.figure(5)
    plt.plot(log['s-PI'])
    plt.plot(log['s-VI'])
    plt.plot(log['s-sarsa'])                                     
    plt.plot(log['s-qlearning'])
    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.title('Example Trajectory for Trained Agents')
    plt.legend(['PI', 'VI', 'SARSA', 'Q-Learning'])
    # plt.plot(log['t'][:-1], log['a'])
    # plt.plot(log['t'][:-1], log['r'])
    plt.savefig('figures/gridworld/trajectories.png')
    plt.show()

if __name__ == '__main__':
    main()
