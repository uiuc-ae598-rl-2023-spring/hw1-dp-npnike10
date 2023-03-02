import random
import numpy as np

class Policy:

    def __init__(self, num_states, num_actions):
        self.policy=np.empty(num_states)
        self.num_actions=num_actions
        self.num_states=num_states

    def initialise(self):
        for s in self.num_states:
            self.policy[s]=random.randrange(self.num_actions)
    
    def epsilon_greedy(self,state,eps,q_table):
        self.eps=eps
        self.q=q_table

        if random.random() < self.eps:
            return random.randrange(self.num_actions)
        else:
            return np.argmax(self.q[state])
    
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
                action=self.pi[st]
                self.v[st]=sum([self.env.p(next_st,st,action)*(self.env.r(st,action)+self.gamma*self.v[next_st]) for next_st in range(self.env.num_states)])
                delta=max(delta,abs(value-self.v[st]))
            if delta<self.theta:
                break
    
    def improve_policy(self):
        for st in range(self.env.num_states):
            self.pi[st]=np.argmax(np.asarray([sum([self.env.p(next_st,st,act)*(self.env.r(st,act)+self.gamma*self.v[next_st]) for next_st in range(self.env.num_states)]) for act in range(self.env.num_actions)]))

    def is_optimal(self,old_pi):
        optimal=True
        for st in range(self.env.num_states):
            if self.pi[st]!=old_pi[st]:
                optimal=False
        return optimal
    
    def learn_policy(self):
        optimal=False
        while not optimal:
            self.evaluate_policy()
            old_pi=self.pi
            self.improve_policy()
            optimal=self.is_optimal(old_pi)
    