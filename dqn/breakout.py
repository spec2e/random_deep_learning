# -*- coding: utf-8 -*-
import random
import argparse
from collections import deque

import gym
import numpy as np

np.random.seed(1337) # for reproducibility

import os

from PIL import Image
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Dense, Convolution2D, Activation, Flatten, Permute
from keras.optimizers import Adam



import time

STEPS = 100000

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

LOG_INTERVAL = 1000

BATCH_SIZE = 32
TRAIN_INTERVAL = 4


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

        optimizer = Adam(lr=.00025)

        model.compile(optimizer=optimizer, loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_to_predict = process_state_batch(state)

        q_values = self.model.predict_on_batch(state_to_predict)

        action = np.argmax(q_values[0])
        return action

    def replay(self):

        start_time = int(round(time.time() * 1000))

        minibatch = random.sample(self.memory, BATCH_SIZE)

        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        terminal1_batch = []

        for state, action, reward, next_state, is_done in minibatch:

            state_batch.append(state[0])
            next_state_batch.append(next_state[0])
            action_batch.append(action)
            reward_batch.append(reward)
            terminal1_batch.append(0. if is_done else 1.)

        #q = self.model.predict(processed_next_state)
        state_batch = np.array(state_batch)
        state_batch = process_state_batch(state_batch)

        next_state_batch = np.array(next_state_batch)
        next_state_batch = process_state_batch(next_state_batch)

        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)

        target_q_values = self.model.predict_on_batch(next_state_batch)

        q_batch = np.max(target_q_values, axis=1).flatten()
        targets = np.zeros((BATCH_SIZE, self.action_size))

        discounted_reward_batch = self.gamma * q_batch

        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch

        Rs = reward_batch + discounted_reward_batch
        for idx, (target, R, action) in enumerate(zip(targets, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward

        targets = np.array(targets).astype('float32')

        loss = self.model.train_on_batch(state_batch, targets )

        return loss

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


def train(args, warmup_steps=5000):

    training = True
    if args['mode'] == 'run':
        agent.epsilon = agent.epsilon_min
        training = False

    """
     The training process will create a state containing 4 images in grayscale from the observation.
     For each step that is taken a next_state is created, which will contain images from the previous three steps
     and the new observation in grayscale.

     To start out the first state will be based on the observation from env.reset and copied three times.
    """

    highscore = 0

    step = 0
    loss = 0

    while step < STEPS:
        agent.current_episode = step

        # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
        state = process_observation(
            env.reset()
        )

        #state = process_state_batch(state)

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

        score = 0

        start_over = True

        lives = 5

        while True:

            if not training:
                env.render()

            start_time = int(round(time.time() * 1000))

            if start_over:
                action =  env.action_space.sample() # start the game
                start_over = False
            else:
                action = agent.act(input_state)


            next_state, reward, is_done, info = env.step(action)

            score += reward

            current_lives = info['ale.lives']
            if current_lives < lives:
                lives = current_lives
                start_over = True

            reward = np.clip(reward, -1, 1)

            # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
            next_state = process_observation(next_state)

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

            if training:
                # Add the beginning state of this action and the outcome to the replay memory
                # if reward != 0.0:
                agent.remember(input_state, action, reward, input_next_state, is_done)

            # Move forward...
            input_state = input_next_state

            step += 1

            if step > LOG_INTERVAL:
                is_done = True

            #if step % TRAIN_INTERVAL == 0:
                #print('step: ', step)

            # If the game has stopped, sum up the result and continue to next episode
            if is_done:
                if score > highscore:
                    highscore = score


            if step % LOG_INTERVAL == 0:
                print('')
                print("step: {}/{}, score: {}, highscore: {}, steps: {}, e: {}, loss: {}"
                      .format(step, STEPS, score, highscore, step, agent.epsilon, loss))


            # If we have remembered observations that exceeds the batch_size (32), we should replay them.
            if step > warmup_steps and step % TRAIN_INTERVAL == 0:
                loss = agent.replay()

            end_time = int(round(time.time() * 1000))
            #print('step-time = ', (end_time - start_time))

            if is_done:
                break

        if training and step > 0 and step % LOG_INTERVAL == 0:
            print('Saving model....')
            agent.save("../save/breakout-dqn-v2.h5")
            print('done!')

        agent.decrease_explore_rate()




    if training:
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
    processed_observation = processed_observation.astype('uint8')  # saves storage in experience memory

    return processed_observation


def process_state_batch(state):
    processed_batch = state
    processed_batch = processed_batch.astype('float32') / 255.
    return processed_batch

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
    #agent.load("../save/dqn_Breakout-v0_weights.h5f")

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())

    train(args)
