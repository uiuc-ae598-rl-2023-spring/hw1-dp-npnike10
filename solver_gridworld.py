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
    
    def learn_policy(self,log):
        optimal=False
        mean_value=np.mean(np.asarray(self.v))
        log['mean-value-PI'].append(mean_value)
        while not optimal:
            self.evaluate_policy()
            old_pi=self.pi.policy.copy()
            self.improve_policy()
            optimal=self.is_optimal(old_pi)

            log['iter-PI'].append(log['iter-PI'][-1] + 1) #TODO check should this be inside evaluate_policy?
            mean_value=np.mean(np.asarray(self.v))
            log['mean-value-PI'].append(mean_value)

        return self.v, self.pi

class ValueIteration:

    def __init__(self,theta,gamma,env):
        self.theta=theta
        self.gamma=gamma
        self.env=env
        self.v=random.sample(range(-100,100), k=env.num_states)
        self.pi=Policy(env.num_states, env.num_actions)
    
    def update_value(self,log):
        mean_value=np.mean(np.asarray(self.v))
        log['mean-value-VI'].append(mean_value)
        while True:
            delta=0
            for st in range(self.env.num_states):
                value=self.v[st]
                action=self.pi.get_action(st)
                self.v[st]=max([sum([self.env.p(next_st,st,action)*(self.env.r(st,action)+self.gamma*self.v[next_st]) for next_st in range(self.env.num_states)]) for act in range(self.env.num_actions)])
                delta=max(delta,abs(value-self.v[st]))
            log['iter-VI'].append(log['iter-VI'][-1] + 1)
            mean_value=np.mean(np.asarray(self.v))
            log['mean-value-VI'].append(mean_value)
            if delta<self.theta:
                break
    
    def learn_policy(self,log):
        self.update_value(log)
        for s in range(self.env.num_states):
            self.pi.greedy(s,self.v,self)
        return self.v, self.pi

class SARSA:

    def __init__(self, theta, gamma, alpha, eps, num_episodes, env):
        self.theta=theta
        self.gamma=gamma
        self.eps=eps
        self.alpha=alpha
        self.num_episodes=num_episodes
        self.env=env
        self.q=np.zeros((env.num_states,env.num_actions))
        self.pi=Policy(env.num_states, env.num_actions)

    def single_episode(self):
        s=self.env.reset()
        a=self.pi.epsilon_greedy(s,self.eps,self.q)
        done=False
        while not done:
            next_s, r, done=self.env.step(a)
            next_a=self.pi.epsilon_greedy(next_s,self.eps,self.q)
            self.q[s][a]+=self.alpha*(r+self.gamma*self.q[next_s][next_a]-self.q[s][a])
            s=next_s
            a=next_a
            
    def learn_policy(self):
        for episode in range(self.num_episodes):
            self.single_episode()
        for s in range(self.env.num_states):
            self.pi.greedy(s,self.q)
        v_TD0=TD0(self.pi,self.alpha,self.gamma,self.num_episodes,self.env)
        values=v_TD0.learn_values()
        return values, self.pi

class QLearning:

    def __init__(self, theta, gamma, alpha, eps, num_episodes, env):
        self.theta=theta
        self.gamma=gamma
        self.eps=eps
        self.alpha=alpha
        self.num_episodes=num_episodes
        self.env=env
        self.q=np.zeros((env.num_states,env.num_actions))
        self.pi=Policy(env.num_states, env.num_actions)

    def single_episode(self):
        s=self.env.reset()
        done=False
        while not done:
            a=self.pi.epsilon_greedy(s,self.eps,self.q)
            next_s, r, done=self.env.step(a)
            next_a=np.argmax(self.q[next_s])
            self.q[s][a]+=self.alpha*(r+self.gamma*self.q[next_s][next_a]-self.q[s][a])
            s=next_s 

    def learn_policy(self):
        for episode in range(self.num_episodes):
            self.single_episode()
        for s in range(self.env.num_states):
            self.pi.greedy(s,self.q)
        v_TD0=TD0(self.pi,self.alpha,self.gamma,self.num_episodes,self.env)
        values=v_TD0.learn_values()
        return values, self.pi

class TD0:

    def __init__(self,policy,alpha,gamma,num_episodes,env): # policy here is a Policy object
        self.alpha=alpha
        self.num_episodes=num_episodes
        self.pi=policy
        self.gamma=gamma
        self.env=env
        self.v=np.zeros(self.env.num_states)
    
    def single_episode(self):
        s=self.env.reset()
        done=False
        while not done:
            a=self.pi.get_action(s)
            next_s, r, done=self.env.step(a)
            self.v[s]+=self.alpha*(r+self.gamma*self.v[next_s]-self.v[s])
            s=next_s

    def learn_values(self):
        for episode in range(self.num_episodes):
            self.single_episode()
        return self.v


    
