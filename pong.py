import numpy as np
import _pickle as pickle
import gym
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    action='store',
                    dest='res',
                    default='True',
                    help='continue from previous checkpoint')
parser.add_argument('--render',
                    action='store',
                    dest='ren',
                    default='False',
                    help='render the screen')
parser.add_argument('--hidden',
                    action='store',
                    dest='hidden',
                    default='128',
                    help='Number of neurons in hidden layer')
parser.add_argument('--batch',
                    action='store',
                    dest='batch',
                    default='16',
                    help='Batch Size')
parsed = parser.parse_args()

# defining hyperparameters

"""
    hyperparameter
    H <- Hidden layer neurons
    b_size <- Batch size
    l_r <- Learning rate
    Y <- (gamma) discount factor for rewards
    decay <- decay factor for RMSProp
    resume <- restart from checkpoint
    render
"""

H = int(parsed.hidden)
b_size = int(parsed.batch)
l_r = 1e-3
Y = 0.99
decay = 0.99
resume = True if parsed.res == 'True' else False
render = False if parsed.ren == 'False' else True

# Model initialization

D = 80 * 80  # Input size [ 80 x 80 ]

if resume:
    model = pickle.load(open('model.p', 'rb'))
    print("Using loaded weights")
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # Xavier initializatuion
    model['W2'] = np.random.randn(H) / np.sqrt(H)

g_buffers = {k: np.zeros_like(v) for k, v in model.items()}  # Buffers to add up grads
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(img):
    """
    preprocessing the image to an 80x80 vector
    """
    img = img[35:195]  # crop
    img = img[::2, ::2, 0]  # downsample (sampling at step = 2)
    img[img == 144] = 0  # erasing the background
    img[img == 109] = 0  # erasing the background
    img[img != 0] = 1  # make everything else 1
    return img.astype(np.float).ravel()


def discount_rewards(r):
    """
    takes 1D float of reward and compute discounted reward
    """
    disc_r = np.zeros_like(r)

    running_add = 0

    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, game boundary
        running_add = running_add * Y + r[t]
        disc_r[t] = running_add
    return disc_r


def policy_fwd(x):
    """
    Forward pass
    """
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return prob and prev hidden state


def policy_back(eph, epdlogp):
    """
    Backward pass
    """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[dh <= 0] = 0  # ReLU
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")   # Initializing the environment

observation = env.reset()
prev_x = None  # diff frame
xs, hs, dlogps, drs = [], [], [], []
running_rew = None
rew_sum = 0
ep_num = 0

# Main loop
while True:

    if render:
        env.render()

    # Preprocess the observations, set i/o to network to be diffrence image

    cur_x = preprocess(observation)

    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # perform forward pass and sample an action from the returned prob
    aprob, h = policy_fwd(x)
    action = 2 if np.random.uniform() < aprob else 3  # pick on random

    # record intermediates
    xs.append(x)  # observations
    hs.append(h)  # hidden states

    y = 1 if action == 2 else 0  # a 'fake label'
    dlogps.append(y - aprob)  # grad to encourage the action that was taken to be taken

    # step in the environment and get new measurements
    observation, reward, done, info = env.step(action)
    rew_sum += reward

    drs.append(reward)  # log the rewards

    if done:
        ep_num += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        xs, hs, dlogps, drs = [], [], [], []  # Reset the arrays

        # compute discounted rewards backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normals
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_back(eph, epdlogp)

        for k in model:
            g_buffers[k] += grad[k]  # accumulate grads over batch

        # performing rmsprop update every batch-sized episodes
        if ep_num % b_size == 0:
            for k, v in model.items():
                g = g_buffers[k]  # gradient
                rmsprop_cache[k] = decay * rmsprop_cache[k] + (1 - decay) * g**2
                model[k] += l_r * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                g_buffers[k] = np.zeros_like(v)  # reset / zero_grads

        # book-keeping

        running_rew = rew_sum if running_rew is None else running_rew * 0.99 + rew_sum * 0.01
        print(f"resetting environment. episode rewards total was: {rew_sum} running mean: {running_rew}")
        if ep_num % 50 == 0:
            pickle.dump(model, open('model.p', 'wb'))
        rew_sum = 0
        observation = env.reset()
        prev_x = None

    if reward != 0:  # +1/-1 reward for each game
        print(f"ep: {ep_num} game finished, reward: {reward}", " " if reward == -1 else "!!!!")
