import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

#/Users/niket/AE598/hw1-dp-npnike10-main


def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Initialize simulation
    s = env.reset()

    # parameters and constants
    theta=0.0005
    gamma=0.95
    alpha=0.5
    eps=0.3
    num_episodes=100000

    # value functions and policies
    old_v=random.sample(range(-100,100),k=env.num_states)
    v_opt={'pi':None, 'vi':None}
    pi_opt={'pi':None, 'vi':None}
    #new_v=np.empty(env.num_states)
    pi={k:random.randrange(env.num_actions) for k in range(env.num_states)}

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    ## policy iteration

    while True:
        # in-place policy evaluation
        while True:
            delta=0
            for st in range(env.num_states):
                v=old_v[st]
                action=pi[st]
                old_v[st]=sum([env.p(s1,st,action)*(env.r(st,action)+gamma*old_v[s1]) for s1 in range(env.num_states)])
                delta=max(delta,abs(v-old_v[st]))
            if delta<theta:
                break
        
        # policy improvement
        optimal=True
        for st in range(env.num_states):
            action=pi[st]
            pi[st]=np.argmax(np.asarray([sum([env.p(s1,st,act)*(env.r(st,act)+gamma*old_v[s1]) for s1 in range(env.num_states)]) for act in range(env.num_actions)]))
            if pi[st]!=action:
                optimal=False

        if optimal:
            v_opt['pi']=old_v
            pi_opt['pi']=pi
            break
    
    print('Policy Iteration:',v_opt['pi'],pi_opt['pi'])

    # value function
    vi_v=random.sample(range(-100,100),k=env.num_states)
    
    ## value iteration
    
    while True:
        delta=0
        for st in range(env.num_states):
            v=vi_v[st]
            action=pi[st]
            
            vi_v[st]=max([sum([env.p(s1,st,act)*(env.r(st,act)+gamma*vi_v[s1]) for s1 in range(env.num_states)]) for act in range(env.num_actions)])
            #print(v,old_v[st])
            delta=max(delta,abs(v-vi_v[st]))
        #print(itr2,delta)

        if delta<theta:
            break

    v_opt['vi']=vi_v   
    vi_pi={k:np.argmax(np.asarray([sum([env.p(s1,k,act)*(env.r(k,act)+gamma*vi_v[s1]) for s1 in range(env.num_states)]) for act in range(env.num_actions)])) for k in range(env.num_states)} 
    pi_opt['vi']=vi_pi

    print('Value Iteration:',v_opt['vi'],pi_opt['vi'])

    ## Sarsa

    def eps_greedy(eps,optimal_act):
        if random.random() < 1-eps:
            return optimal_act
        else:
            return random.randrange(env.num_actions)
    
    # value functions and policies
    sarsa_q={st:{a:random.uniform(-100,100) for a in range(env.num_actions)} for st in range(env.num_states)}

    #learning
    for episode in range(num_episodes):
        env.reset()
        opt_a=np.argmax(np.asarray([sarsa_q[env.s][act] for act in range(env.num_actions)]))
        a=eps_greedy(eps,opt_a)
        done=False
        while not done:
            next_s, r, done=env.step(a)
            next_a=eps_greedy(eps,opt_a)
            sarsa_q[env.s][a]+=alpha*(r+gamma*sarsa_q[next_s][next_a]-sarsa_q[env.s][a])

            env.s=next_s
            a=next_a

    sarsa_pi={st:np.argmax(np.asarray([sarsa_q[st][act] for act in range(env.num_actions)])) for st in range(env.num_states)}

    print('SARSA', sarsa_pi)
            
    ## Q-learning

    # value functions
    ql_q={st:{a:random.uniform(-100,100) for a in range(env.num_actions)} for st in range(env.num_states)}

    #learning
    for episode in range(num_episodes):
        env.reset()
        done=False
        while not done:
            opt_a=np.argmax(np.asarray([ql_q[env.s][act] for act in range(env.num_actions)]))
            a=eps_greedy(eps,opt_a)
            next_s, r, done=env.step(a)
            ql_q[env.s][a]+=alpha*(r+gamma*max([ql_q[next_s][act] for act in range(env.num_actions)])-ql_q[env.s][a])

            env.s=next_s

    ql_pi={st:np.argmax(np.asarray([ql_q[st][act] for act in range(env.num_actions)])) for st in range(env.num_states)}

    print('Q-Learning', ql_pi)

            


    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(env.num_actions)
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
