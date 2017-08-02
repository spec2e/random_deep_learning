# -*- coding: utf-8 -*-
import random
from collections import deque

import gym
import numpy as np

np.random.seed(1337) # for reproducibility

import os

from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, Flatten, Permute
from keras.optimizers import Adam


EPISODES = 100000

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

TRAIN_SIZE = 100

BATCH_SIZE = 32


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.gamma = 0.95  # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.001
        self.epsilon = self.epsilon_max
        self.current_episode = 0
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.train_queue = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=state_size))
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])
        return action

    def replay(self):
        train_queue = []

        for count in range(TRAIN_SIZE):
            #print('count: ', count)
            minibatch = random.sample(self.memory, BATCH_SIZE)
            mse = 0

            for state, action, reward, next_state, is_done in minibatch:

                processed_state = process_state(state)
                processed_next_state = process_state(next_state)

                q_for_next_state = self._calculate_Q_for_next_state(is_done, processed_next_state, reward)
                #print('q value: ', q_for_next_state)
                target_t = self.model.predict(processed_state)
                target_t[0][action] = q_for_next_state

                train_queue.append((
                    processed_state,
                    target_t
                ))


        print('now training')

        x_train = []
        y_train = []

        for train_state, train_target in train_queue:
            x_train.append(train_state)
            y_train.append(train_target)

        self.model.train_on_batch(
            x_train[0],
            y_train[0]
        )

        #mse += (q_for_next_state - target_t[0][action]) ** 2
        #print('mse: ', mse)



    def decrease_explore_rate(self):
        # Linear annealed: f(x) = ax + b.
        a = -float(self.epsilon_max - self.epsilon_min) / float(50000)
        b = float(self.epsilon_max)
        value = a * float(self.current_episode) + b
        self.epsilon = max(self.epsilon_min, value)

    def _calculate_Q_for_next_state(self, is_done, next_state, reward):
        result_of_next_state = reward
        if not is_done:
            next_state_prediction = self.model.predict(next_state)
            q_prediction = np.max(
                next_state_prediction[0]
            )

            discounted_reward = self.gamma * q_prediction
            target_reward = reward + discounted_reward


            #if reward == 1 or reward > 1:
            #    print('----------------------')
            #    print('reward: ', reward)
            #    print('target_reward: ', target_reward)


            result_of_next_state = target_reward

        return result_of_next_state

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name, overwrite=True)

    def print_memory(self):
        print(len(self.memory))


def train():
    """
     The training process will create a state containing 4 images in grayscale from the observation.
     For each step that is taken a next_state is created, which will contain images from the previous three steps
     and the new observation in grayscale.

     To start out the first state will be based on the observation from env.reset and copied three times.
    """
    highscore = 0

    for e in range(EPISODES):

        agent.current_episode = e

        # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
        state = process_observation(
            env.reset()
        )

        state = process_state(state)

        # reshape the state to fit input to the convolutional network. The dimensions must be 4 x 84 x 84.
        # That is 4 images in grayscale with 84 x 84 pixels
        input_state = reshape_to_fit_network(
            np.stack(
                (
                    state,
                    state,
                    state,
                    state
                )
            )
        )

        lives = 5

        score = 0
        step = 0
        while True:

            #env.render()

            action = agent.act(input_state)

            next_state, reward, is_done, info = env.step(action)

            score += reward

            """
            current_lives = info['ale.lives']

            if current_lives < lives:
                lives = current_lives
                reward = -1
            """

            #reward = np.clip(reward, -1, 1)

            # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
            next_state = process_observation(next_state)
            next_state = process_state(next_state)

            # reshape the state to fit input to the convolutional network. The dimensions must be 4 x 84 x 84.
            # That is 4 images in grayscale with 84 x 84 pixels
            # Swap out the first image and replace with the former next_state
            input_next_state = reshape_to_fit_network(
                np.stack(
                    (
                        input_state[0][1],
                        input_state[0][2],
                        input_state[0][3],
                        next_state
                    )
                )
            )

            # Add the beginning state of this action and the outcome to the replay memory
            # if reward != 0.0:
            agent.remember(input_state, action, score, input_next_state, is_done)

            # Move forward...
            input_state = input_next_state

            step += 1

            # If the game has stopped, sum up the result and continue to next episode
            if is_done:
                if score > highscore:
                    highscore = score

                print("episode: {}/{}, score: {}, highscore: {}, steps: {}, e: {}, memory_length: {}"
                      .format(e, EPISODES, score, highscore, step, agent.epsilon, len(agent.memory)))
                break

        # If we have remembered observations that exceeds the batch_size (32), we should replay them.
        if e > 0 and e % TRAIN_SIZE == 0:
            print('replaying steps...')
            agent.replay()
            print('Saving model....')
            agent.save("../save/breakout-dqn-v2.h5")
            print('done!')

        agent.decrease_explore_rate()

    agent.replay()
    print('Saving final model....')
    agent.save("../save/breakout-dqn-v2.h5")
    print('done training!')


def reshape_to_fit_network(input):
    return input.reshape(1, input.shape[0], input.shape[1], input.shape[2])  # 1*84*84*4


def save_image(img_array, file_name, folder_name):
    folders = "images/{}".format(folder_name)

    if not os.path.exists(folders):
        os.makedirs(folders)

    img = Image.fromarray(img_array)
    img.save("{}/{}".format(folders, file_name))


def process_observation(observation):
    assert observation.ndim == 3  # (height, width, channel)
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale

    processed_observation = np.array(img)
    assert processed_observation.shape == INPUT_SHAPE
    batch = processed_observation.astype('uint8')  # saves storage in experience memory

    return batch

def process_state(state):

    processed_batch = state.astype('float32') / 255.
    return processed_batch



def play_game():

    agent.epsilon = 0.05

    highscore = 0

    for e in range(EPISODES):

        # Get the first observation and make it grayscale and reshapeto 84 x 84 pixels
        state = process_observation(
            env.reset()
        )

        # reshape the state to fit input to the convolutional network. The dimensions must be 4 x 84 x 84.
        # That is 4 images in grayscale with 84 x 84 pixels
        input_state = reshape_to_fit_network(
            np.stack(
                (
                    state,
                    state,
                    state,
                    state
                )
            )
        )

        start_over = True

        lives = 5

        score = 0

        for step in range(5000):

            env.render()

            if start_over:
                action = 1  # start the game
                start_over = False
            else:
                action = agent.act(input_state)

            next_state, reward, done, info = env.step(action)

            score += reward

            current_lives = info['ale.lives']
            if current_lives < lives:
                lives = current_lives
                start_over = True

            # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
            next_state = process_observation(next_state)
            # save_image(input_state[0][0], 'input1_img.png', str(step))
            # save_image(input_state[0][1], 'input2_img.png', str(step))
            # save_image(input_state[0][2], 'input3_img.png', str(step))
            # save_image(next_state, 'next_state_img.png', str(step))

            # reshape the state to fit input to the convolutional network. The dimensions must be 4 x 84 x 84.
            # That is 4 images in grayscale with 84 x 84 pixels
            # Swap out the first image and replace with the former next_state
            input_next_state = reshape_to_fit_network(
                np.stack(
                    (
                        input_state[0][1],
                        input_state[0][2],
                        input_state[0][3],
                        next_state
                    )
                )
            )

            # Move forward...
            input_state = input_next_state

            # If the game has stopped, sum up the result and continue to next episode
            if done:
                if score > highscore:
                    highscore = score

                print("episode: {}/{}, score: {}, highscore: {}, steps: {}"
                      .format(e, EPISODES, score, highscore, step))
                break



class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v0')
    state_size = (WINDOW_LENGTH,) + INPUT_SHAPE
    print('state_size: ', state_size)
    action_size = env.action_space.n
    print('action size')
    print(action_size)
    agent = DQNAgent(state_size, action_size)
    done = False
    #agent.load("../save/breakout-dqn.h5")
    #agent.load("../save/breakout-dqn-v2.h5")
    #play_game()
    train()
