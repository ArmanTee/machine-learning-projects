import gym
env = gym.make('Blackjack-v0')
print(env.observation_space)
print(env.action_space)
for i_episode in range(3):
state = env.reset()
while True:
    print(state)
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        print('End game! Reward: ', reward)
        print('You won :)\n') if reward > 0 else print('You lost :(\n')
        break

def generate_episode_from_limit(bj_env):
episode = []
state = bj_env.reset()
while True:
    action = 0 if state[0] > 18 else 1
    next_state, reward, done, info = bj_env.step(action)
    episode.append((state, action, reward))
    state = next_state
    if done:
        break
return episode
([1,2,3]).pop()
list1 = ([1,2,3])
list1[:-1]
list1.remove(len(list1))
del(list1[-1])
a= 0
a
list1
for i in range(20):
print(generate_episode_from_limit(env))
episode  = generate_episode_from_limit(env)
len(episode)
set(episode[:][0])
set(np.array(episode)[:,0])
list(distincts.index(i) for i in distincts)
list(distincts).index()
for i,j in enumerate(((np.array(episode)[:,0]))):
 print(i)
 print(j)
from collections import defaultdict
import numpy as np
import sys
episode.index()
sum(np.array(episode)[1:,2])
episode[1:,1]
list(np.array(episode)[:,0])
def mc_prediction_v(env, num_episodes, generate_episode, gamma=1.0):
# initialize empty dictionary of lists
returns = defaultdict(lambda:np.zeros(num_episodes))
N = defaultdict(lambda:0)
# loop over episodes
for i_episode in range(1, num_episodes+1):
    # monitor progress
    if i_episode % 1000 == 0:
        print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        sys.stdout.flush()
    episode = generate_episode_from_limit(env)
    distincts = list(set(np.array(episode)[:,0]))
    for s,s_names in enumerate(distincts[:-1]):
        N[s_names]+= 1
        ind = list(np.array(episode)[:,0]).index(s_names)
        returns[s_names] = sum(np.array(episode)[ind:,2])


    ## TODO: complete the function

return N,returns


1000

from plot_utils import plot_blackjack_values
gamma
# obtain the value function
V = mc_prediction_v(env, 500000, generate_episode_from_limit)
V
from pd.DataFrame(V)
pd.dataFrame
/V[1]

########### Actual Function ######################
from collections import defaultdict
import numpy as np
import sys

def mc_prediction_v(env, num_episodes, generate_episode, gamma=1.0):
# initialize empty dictionary of lists
returns = defaultdict(list)
# loop over episodes
for i_episode in range(1, num_episodes+1):
    # monitor progress
    if i_episode % 1000 == 0:
        print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        sys.stdout.flush()
    # generate an episode
    episode = generate_episode(env)
    # obtain the states, actions, and rewards
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    # calculate and store the return for each visit in the episode
    for i, state in enumerate(states):
        returns[state].append(sum(rewards[i:]*discounts[:-(1+i)]))
# calculate the state-value function estimate
V = {k: np.mean(v) for k, v in returns.items()}
return V



from plot_utils import plot_blackjack_values

# obtain the value function
V = mc_prediction_v(env, 500000, generate_episode_from_limit)

# plot the value function
plot_blackjack_values(V)


######## STOCHASTIC EPISODE GENERATION ###########

### Play the game #####
def generate_episode_from_limit_stochastic(bj_env):
episode = []
state = bj_env.reset()
while True:
    probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
    action = np.random.choice(np.arange(2), p=probs)
    next_state, reward, done, info = bj_env.step(action)
    episode.append((state, action, reward))
    state = next_state
    if done:
        break
return episode


### Q prediction ########

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
# initialize empty dictionaries of arrays
returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
N = defaultdict(lambda: np.zeros(env.action_space.n))
Q = defaultdict(lambda: np.zeros(env.action_space.n))
# loop over episodes
for i_episode in range(1, num_episodes+1):
    # monitor progress
    if i_episode % 1000 == 0:
        print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        sys.stdout.flush()
    # generate an episode
    episode = generate_episode(env)
    # obtain the states, actions, and rewards
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    # update the sum of the returns, number of visits, and action-value
    # function estimates for each state-action pair in the episode
    for i, state in enumerate(states):
        returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
        N[state][actions[i]] += 1.0
        Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
return Q


# obtain the action-value function
Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

# obtain the state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
     for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V_to_plot)

#################### GLIE  ########################
def generate_episode_from_Q(env, Q, epsilon, nA):
""" generates an episode from following the epsilon-greedy policy """
episode = []
state = env.reset()
while True:
    action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                if state in Q else env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    episode.append((state, action, reward))
    state = next_state
    if done:
        break
return episode

def get_probs(Q_s, epsilon, nA):
""" obtains the action probabilities corresponding to epsilon-greedy policy """
policy_s = np.ones(nA) * epsilon / nA
best_a = np.argmax(Q_s)
policy_s[best_a] = 1 - epsilon + (epsilon / nA)
return policy_s

def update_Q_GLIE(env, episode, Q, N, gamma):
""" updates the action-value function estimate using the most recent episode """
states, actions, rewards = zip(*episode)
# prepare for discounting
discounts = np.array([gamma**i for i in range(len(rewards)+1)])
for i, state in enumerate(states):
old_Q = Q[state][actions[i]]
old_N = N[state][actions[i]]
Q[state][actions[i]] = old_Q + (sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)/(old_N+1)
N[state][actions[i]] += 1
return Q, N



def mc_control_GLIE(env, num_episodes, gamma=1.0):
nA = env.action_space.n
# initialize empty dictionaries of arrays
Q = defaultdict(lambda: np.zeros(nA))
N = defaultdict(lambda: np.zeros(nA))
# loop over episodes
for i_episode in range(1, num_episodes+1):
# monitor progress
if i_episode % 1000 == 0:
    print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
    sys.stdout.flush()
# set the value of epsilon
epsilon = 1.0/((i_episode/8000)+1)
# generate an episode by following epsilon-greedy policy
episode = generate_episode_from_Q(env, Q, epsilon, nA)
# update the action-value function estimate using the episode
Q, N = update_Q_GLIE(env, episode, Q, N, gamma)
# determine the policy corresponding to the final action-value function estimate
policy = dict((k,np.argmax(v)) for k, v in Q.items())
return policy, Q


# obtain the estimated optimal policy and action-value function
policy_glie, Q_glie = mc_control_GLIE(env, 500000)

# obtain the state-value function
V_glie = dict((k,np.max(v)) for k, v in Q_glie.items())

# plot the state-value function
plot_blackjack_values(V_glie)

from plot_utils import plot_policy

# plot the policy
plot_policy(policy_glie)



def update_Q_alpha(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q



def mc_control_alpha(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = 1.0/((i_episode/8000)+1)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q_alpha(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q


# obtain the estimated optimal policy and action-value function
policy_alpha, Q_alpha = mc_control_alpha(env, 500000, 0.02)
