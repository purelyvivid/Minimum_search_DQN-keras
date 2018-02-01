
# coding: utf-8

# - ref: https://github.com/kuangliu/pytorch-cifar
# - ref: https://morvanzhou.github.io/tutorials/machine-learning/torch/4-05-DQN/


import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error as MSE
from keras.losses import categorical_crossentropy as NLL 
from time import time


# # Hyperparameter


BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # Optimal action selection ratio
GAMMA = 0.9                 # Reward decay coefficient
TARGET_REPLACE_ITER = 100   # Q_target update frequency
MEMORY_CAPACITY = 300       # memory capacity
action_space = [ '+', '-' ]     # Just two action: go right and go left
N_ACTIONS = len(action_space)
state_space = [ s for s in range(0,10) ] # 10 states, from interger 0 to 10
N_STATES = len(state_space)


# # Environment

# create an environment, and assume it's unknown



class Environment(object):
    def __init__(self, ):
        self.argmin = 0
        while (self.argmin<=0 or self.argmin>=N_STATES-1):  #Avoid the boundary min.
            self.hidden_coeff = np.random.randint(N_STATES,2*N_STATES,2)
            self.model = self._build_model()
            self.arr = np.array([ self.model(x) for x in range(N_STATES) ])
            self.argmin = np.argmin(self.arr)
        self.min = np.min(self.arr).astype('float32')
        self.stop_condition = self.min + 0.001*np.abs(self.min)/float(N_STATES)
        print('build_model: hidden_coeff={}, argmin={}, min={}, stop_condition={} '.format(self.hidden_coeff,self.argmin, self.min, self.stop_condition))
        self.current_state = np.random.randint(0,N_STATES)
         
    def _build_model(self,):
        def fn(x):
            scale, bias = self.hidden_coeff
            op = np.sin(scale*float(x)*N_STATES + bias) 
            return op
        return fn
             
    def reset(self, ): 
        self.current_state = np.random.randint(0,N_STATES)
        print ('reset:  current_state={}'.format(self.current_state))
        return self.current_state
    
        
    def step(self, action): # define reward
        s = self.current_state
        y = self.model(s)
        if action == 0:
            s+= 1 # go to right
        else:
            s-= 1 # go to left
        self.current_state = s
        y_ = self.model(s)
        s_ = self.current_state
        r = (y-y_)
        boo_reach_min = (y <= self.stop_condition)
        boo_out_of_range = (s_ < 0 or s_>= N_STATES) # if out of range
        done = False
        if  boo_reach_min: 
            r *= 50
            done = True
        else:
            if boo_out_of_range:
                r *= 100 if r < 0 else -100
                done = True
        info = None
        print ('step: y={:.4f}, y_={:.4f}, next_state={}, reward={:.4f}, done={}'.format(y, y_, self.current_state,r,done))
        return s_, r, done, info
        




env= Environment() # retry until the evironment is satisfied 




plt.plot(range(N_STATES), env.arr)


# # Agent



class Net():
    def __init__(self, ):
        self.model = self.build_model( input_shape=(N_STATES,) )
        self.model.compile(optimizer=SGD(LR), loss=MSE)
        
    def build_model(self, input_shape):
        model = Sequential([
            Dense(N_STATES*2, input_shape=input_shape),
            Activation('relu'),
            Dense(N_ACTIONS),
        ])  
        #print(model.summary())
        return model
    
    def train(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def predict(self, x):
        return self.model.predict_on_batch(x)

    

net = Net()# for testing




class DQN(object):
    def __init__(self):
        self.target_net , self.eval_net = Net(),  Net() #A Target Net and an Eval Net.
        self.learn_step_counter = 0     # used when Q_target updates
        self.memory_counter = 0         # count current memory state 
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))     # initialize memory 
        self.eval_loss_histroy = []

    def choose_action(self, x): # 'x' is a list, means state, e.g. [0,1,0,0,0,...,0]
        x = np.array([x]) # => 2-D array , here, we just input ONE sample
        s = np.argmax(x)  # a scalar
        if np.random.uniform() < EPSILON:   # if choose the optimal action
            actions_value = self.eval_net.predict(x)
            action = np.argmax(actions_value)  # return the argmax
            print('choose_action: {} from state {}'.format(action, s))
        else:   # if choose a random action
            action = np.random.randint(0, N_ACTIONS)
            print('choose_action: {} randomly from state {}'.format(action, s))
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # if memory is full, rewrite the old data
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.model.set_weights(self.eval_net.model.get_weights())
        self.learn_step_counter += 1
        self.memory_counter = 0

        # extract batch data from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :N_STATES]                         # shape (BATCH_SIZE, N_STATES)
        b_a = b_memory[:, N_STATES:N_STATES+1].astype(int)   # shape (BATCH_SIZE, 1)
        b_r = b_memory[:, N_STATES+1:N_STATES+2]             # shape (BATCH_SIZE, 1)
        b_s_ = b_memory[:, -N_STATES:]                       # shape (BATCH_SIZE, N_STATES)
        assert b_s.shape == (BATCH_SIZE, N_STATES)
        assert b_a.shape == (BATCH_SIZE, 1)
        assert b_r.shape == (BATCH_SIZE, 1)
        assert b_s_.shape == (BATCH_SIZE, N_STATES)

        # q_eval has values of all actions, 
        # but we just take the values of action from vector b_a
        # replace them with the values of Q_target
        q_eval = self.eval_net.predict(b_s) # shape (BATCH_SIZE, N_ACTIONS)
        q_next = self.target_net.predict(b_s_)    # shape (BATCH_SIZE, N_ACTIONS), q_next 不进行反向传递误差
        assert q_eval.shape == (BATCH_SIZE, N_ACTIONS)
        assert q_next.shape == (BATCH_SIZE, N_ACTIONS)
        q_target = q_eval # shape (BATCH_SIZE, N_ACTIONS)
        max_ = np.expand_dims(np.max(q_next, axis=1),1)
        assert max_.shape == (BATCH_SIZE, 1)
        q_ = b_r + GAMMA * max_ 
        assert q_.shape == (BATCH_SIZE, 1)
        q_target[range(BATCH_SIZE),b_a] =  q_ # shape (BATCH_SIZE, 1)

        # update eval net
        loss = self.eval_net.train(b_s, q_target)
        self.eval_loss_histroy.append(loss)
        
    def plot_loss(self, ):
        print(self.eval_loss_histroy)
        plt.plot( range(len(self.eval_loss_histroy)) , self.eval_loss_histroy)
        plt.show()
        

dqn = DQN()       




#total_history = []
for i_episode in range(100000):
    #epi_history = []
    print('\nepisode: {}'.format(i_episode))
    s = env.reset()
    while True:
        x = [1 if i==s else 0 for i in range(N_STATES)]
        a = dqn.choose_action(x)
        #epi_history.append((s, a))

        # choose action and get reward from environment
        s_, r, done, info = env.step(a)
       
        # save memory
        x_ = [1 if i==s_ else 0 for i in range(N_STATES)]
        dqn.store_transition(x, a, r, x_)  #(s, a, r, s_)

        if dqn.memory_counter >  MEMORY_CAPACITY :
            dqn.learn() # if memory is full, learn
            dqn.plot_loss()

        if done:    
            print ("episode end at {}".format(s))
            break

        s = s_
    #total_history.append(epi_history)











