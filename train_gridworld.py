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
    eps=0.3
    alpha=0.1
    num_episodes=5000

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
        'returns-qlearning': []
    }

    log1={'returns-sarsa': [],
        'returns-qlearning': []}

    log2={'returns-sarsa': [],
        'returns-qlearning': []}
    
    log3={'returns-sarsa': [],
        'returns-qlearning': []}

    log4={'returns-sarsa': [],
        'returns-qlearning': []}
    
    log5={'returns-sarsa': [],
        'returns-qlearning': []}

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

    logs=[log1,log2,log3,log4,log5]

    def multiple_learning_curves(params,identifier):
        if identifier==0:
            for i, param in enumerate(params):
                QLearning_agent_e=QLearning(theta, gamma, alpha,param , num_episodes, env)
                QLearning_agent_e.learn_policy(logs[i])

                sarsa_agent_e=SARSA(theta, gamma, alpha, param, num_episodes, env)
                sarsa_agent_e.learn_policy(logs[i])
        if identifier==1:
            for i, param in enumerate(params):
                QLearning_agent_e=QLearning(theta, gamma, param, eps, num_episodes, env)
                QLearning_agent_e.learn_policy(logs[i])

                sarsa_agent_e=SARSA(theta, gamma, param, eps, num_episodes, env)
                sarsa_agent_e.learn_policy(logs[i])
        

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
    epsilons=[0.1, 0.3, 0.5, 0.7, 0.9]
    multiple_learning_curves(epsilons,0)
    fig6,(axs1,axs2)=plt.subplots(1,2)
    axs1.plot(range(1,num_episodes+1),log1['returns-sarsa'])
    axs1.plot(range(1,num_episodes+1),log2['returns-sarsa'])
    axs1.plot(range(1,num_episodes+1),log3['returns-sarsa'])
    axs1.plot(range(1,num_episodes+1),log4['returns-sarsa'])
    axs1.plot(range(1,num_episodes+1),log5['returns-sarsa'])  
    axs2.plot(range(1,num_episodes+1),log1['returns-qlearning'])
    axs2.plot(range(1,num_episodes+1),log2['returns-qlearning'])
    axs2.plot(range(1,num_episodes+1),log3['returns-qlearning'])
    axs2.plot(range(1,num_episodes+1),log4['returns-qlearning'])
    axs2.plot(range(1,num_episodes+1),log5['returns-qlearning']) 
    axs1.set_title('SARSA')                                
    axs2.set_title('Q-Learning')
    fig6.suptitle('Learning Curves for Varying Epsilon Values')
    axs1.set(xlabel='Number of Episodes',ylabel='Return')
    axs2.set(xlabel='Number of Episodes',ylabel='Return')
    axs1.legend(epsilons)
    axs2.legend(epsilons)
    fig6.savefig('figures/gridworld/m_eps_lcs.png')
    fig6.tight_layout()
    plt.show()

    alphas=[0.001, 0.01, 0.1, 0.3, 0.5]
    for l in logs:
        l['returns-sarsa'].clear()
        l['returns-qlearning'].clear()
    multiple_learning_curves(alphas,1)

    fig7,(axs3,axs4)=plt.subplots(1,2)
    axs3.plot(range(1,num_episodes+1),log1['returns-sarsa'])
    axs3.plot(range(1,num_episodes+1),log2['returns-sarsa'])
    axs3.plot(range(1,num_episodes+1),log3['returns-sarsa'])
    axs3.plot(range(1,num_episodes+1),log4['returns-sarsa'])
    axs3.plot(range(1,num_episodes+1),log5['returns-sarsa'])  
    axs4.plot(range(1,num_episodes+1),log1['returns-qlearning'])
    axs4.plot(range(1,num_episodes+1),log2['returns-qlearning'])
    axs4.plot(range(1,num_episodes+1),log3['returns-qlearning'])
    axs4.plot(range(1,num_episodes+1),log4['returns-qlearning'])
    axs4.plot(range(1,num_episodes+1),log5['returns-qlearning']) 
    axs3.set_title('SARSA')                                
    axs4.set_title('Q-Learning')
    fig7.suptitle('Learning Curves for Varying Alpha Values')
    axs3.set(xlabel='Number of Episodes',ylabel='Return')
    axs4.set(xlabel='Number of Episodes',ylabel='Return')
    axs3.legend(alphas)
    axs4.legend(alphas)
    fig7.savefig('figures/gridworld/m_alphas_lcs.png')
    fig7.tight_layout()
    plt.show()

    plt.figure(1)
    plt.plot(log['iter-PI'], log['mean-value-PI'])
    plt.plot(log['iter-VI'], log['mean-value-VI'])
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean of Value Function')
    plt.title('Learning Curve for Model-Based methods')
    plt.legend(['PI','VI'])
    plt.savefig('figures/gridworld/mean-values.png')
    plt.show()


    plt.figure(2)
    plt.plot(PI_policy.policy)
    plt.plot(VI_policy.policy)
    plt.plot(sarsa_policy.policy)
    plt.plot(QLearning_policy.policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title('Policy of the Trained Agents')
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

    fig5,(axs1,axs2)=plt.subplots(2,2,figsize=(12, 10))
    axs1[0].plot(log['s-PI'])
    axs1[0].plot(log['a-PI'])
    axs1[0].plot(log['r-PI'])
    axs1[1].plot(log['s-VI'])
    axs1[1].plot(log['a-VI'])
    axs1[1].plot(log['r-VI'])
    axs2[0].plot(log['s-sarsa'])
    axs2[0].plot(log['a-sarsa'])
    axs2[0].plot(log['r-sarsa'])                                    
    axs2[1].plot(log['s-qlearning'])
    axs2[1].plot(log['a-qlearning'])
    axs2[1].plot(log['r-qlearning'])
    axs1[0].set_title('PI')
    axs1[1].set_title('VI')
    axs2[0].set_title('SARSA')                                
    axs2[1].set_title('Q-Learning')
    for i in range(len(axs1)):
        axs1[i].set(xlabel='Time step',ylabel='State')
        axs1[i].legend(['States','Actions','Rewards'])
    for i in range(len(axs2)):
        axs2[i].set(xlabel='Time step',ylabel='State')
        axs2[i].legend(['States','Actions','Rewards'])
    fig5.suptitle('Example Trajectory for Trained Agents')
    fig5.savefig('figures/gridworld/trajectories.png')
    fig5.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
