import random
import numpy as np

class Policy:

    def __init__(self, num_states, num_actions):
        self.policy=np.empty(num_states)
        self.num_actions=num_actions
        self.num_states=num_states

    def initialise(self):
        for s in range(self.num_states):
            self.policy[s]=random.randrange(self.num_actions)
    
    def epsilon_greedy(self,state,eps,q_table):
        self.eps=eps
        self.q=q_table

        if random.random() < self.eps:
            return random.randrange(self.num_actions)
        else:
            return np.argmax(self.q[state])

    def greedy(self,s,value_table,agent=None):
        if len(np.asarray(value_table).shape)==1:
            v=value_table
            self.policy[s]=np.argmax(np.asarray([sum([agent.env.p(s1,s,act)*(agent.env.r(s,act)+agent.gamma*v[s1]) for s1 in range(agent.env.num_states)]) for act in range(agent.env.num_actions)]))
        else:
            self.q=value_table
            self.policy[s]=np.argmax(self.q[s])
    
    def get_action(self,s):
        return self.policy[s]
    
    def set_action(self,s,a):
        self.policy[s]=a

class PolicyIteration:

    def __init__(self, theta, gamma, env):
        self.theta=theta
        self.gamma=gamma
        self.env=env
        self.v=random.sample(range(-100,100), k=env.num_states)
        self.pi=Policy(env.num_states, env.num_actions)
        self.pi.initialise()

    # in-place policy evaluation
    def evaluate_policy(self):
        while True:
            delta=0
            for st in range(self.env.num_states):
                value=self.v[st]
                action=self.pi.get_action(st)
                self.v[st]=sum([self.env.p(next_st,st,action)*(self.env.r(st,action)+self.gamma*self.v[next_st]) for next_st in range(self.env.num_states)])
                delta=max(delta,abs(value-self.v[st]))
            if delta<self.theta:
                break
    
    def improve_policy(self):
        for st in range(self.env.num_states):
            self.pi.set_action(st,np.argmax(np.asarray([sum([self.env.p(next_st,st,act)*(self.env.r(st,act)+self.gamma*self.v[next_st]) for next_st in range(self.env.num_states)]) for act in range(self.env.num_actions)])))

    def is_optimal(self,old_pi):
        optimal=True
        for st in range(self.env.num_states):
            if self.pi.get_action(st)!=old_pi[st]:
                optimal=False
        return optimal
    
    def learn_policy(self):
        optimal=False
        while not optimal:
            self.evaluate_policy()
            old_pi=self.pi.policy.copy()
            self.improve_policy()
            optimal=self.is_optimal(old_pi)
        return self.v, self.pi

class ValueIteration:

    def __init__(self,theta,gamma,env):
        self.theta=theta
        self.gamma=gamma
        self.env=env
        self.v=random.sample(range(-100,100), k=env.num_states)
        self.pi=Policy(env.num_states, env.num_actions)
    
    def update_value(self):
        while True:
            delta=0
            for st in range(self.env.num_states):
                value=self.v[st]
                action=self.pi.get_action(st)
                self.v[st]=max([sum([self.env.p(next_st,st,action)*(self.env.r(st,action)+self.gamma*self.v[next_st]) for next_st in range(self.env.num_states)]) for act in range(self.env.num_actions)])
                delta=max(delta,abs(value-self.v[st]))
            if delta<self.theta:
                break
    
    def learn_policy(self):
        self.update_value()
        for s in range(self.env.num_states):
            self.pi.greedy(s,self.v,self)
        return self.v, self.pi

class SARSA:

    def __init__(self, theta, gamma, eps, num_episodes, env):
        self.theta=theta
        self.gamma=gamma
        self.eps=eps
        self.num_episodes=num_episodes
        self.env=env
        self.q=np.zeros((env.num_states,env.num_actions))
        self.pi=Policy(env.num_states, env.num_actions)

    
