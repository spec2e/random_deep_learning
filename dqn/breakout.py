# -*- coding: utf-8 -*-
import argparse
import random
from collections import deque

import gym
import numpy as np

from dqn.DQNModel import DQNModel

np.random.seed(1337)  # for reproducibility

import os


from PIL import Image

from keras.callbacks import TensorBoard

import tensorflow as tf

STEPS = 1000000
EPSILON_DECAY_RATE = 500000

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

TARGET_MODEL_UPDATE_RATE = 2500
SAVE_RATE = 10000

LOG_INTERVAL = 1000

BATCH_SIZE = 32
TRAIN_INTERVAL = 8


class DQNAgent:

    def __init__(self, dqn_model):

        self.dqn_model = dqn_model

        self.memory = deque(maxlen=500000)
        self.gamma = 0.95  # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.001
        self.epsilon = self.epsilon_max
        self.current_step = 0
        self.learning_rate = 0.00025
        self.training_model = self.dqn_model.build_training_model()
        self.run_model = self.dqn_model.build_run_model()
        self.target_model = self.dqn_model.build_target_model()
        self.update_counter = 0
        self.training = True
        self.save_state = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.dqn_model.action_size)

        state_to_predict = process_state_batch(state)
        q_values = []

        if self.training:
            q_values = self.target_model.predict_on_batch(state_to_predict)
        else:
            q_values = self.run_model.predict_on_batch(state_to_predict)

        action = np.argmax(q_values[0])

        return action

    def replay(self):

        minibatch = random.sample(self.memory, BATCH_SIZE)

        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        terminal1_batch = []

        self.update_counter += 1

        for state, action, reward, next_state, is_done in minibatch:
            state_batch.append(state[0])
            next_state_batch.append(next_state[0])
            action_batch.append(action)
            reward_batch.append(reward)
            terminal1_batch.append(0. if is_done else 1.)

        state_batch = np.array(state_batch)
        state_batch = process_state_batch(state_batch)

        next_state_batch = np.array(next_state_batch)
        next_state_batch = process_state_batch(next_state_batch)

        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)

        q_batch = self.predict_on_target(next_state_batch)

        # Discount the Q value from the network. Gamma is the discount factor
        discounted_reward_batch = self.gamma * q_batch

        # Set discounted reward to zero for all states that were terminal.
        # If we died in this state, make sure it is zero!
        discounted_reward_batch *= terminal1_batch

        targets = np.zeros((BATCH_SIZE, self.dqn_model.action_size))
        dummy_targets = np.zeros((BATCH_SIZE,))
        masks = np.zeros((BATCH_SIZE, self.dqn_model.action_size))

        # The reward for the current state must be added with the discounted reward for the next state
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            # The target array which has the shape of 1 x 4, will be updated with future reward for the
            # action that led to the future state
            target[action] = R
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action

        # Apparently it helps the network to convert the targets list to the same data type as the images
        # we use use to train the network
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        loss = self.train_on_model(dummy_targets, masks, state_batch, targets)
        # ['loss', 'loss_loss', 'activation_5_loss', 'loss_mean_q', 'activation_5_mean_q']
        summary = tf.Summary(value=[tf.Summary.Value(tag="training_loss",
                                                     simple_value=loss[0]), ])

        self.dqn_model.log_summary(summary, global_step=agent.current_step)

        if self.update_counter > TARGET_MODEL_UPDATE_RATE:
            print('setting weights on target_model...')
            self.target_model.set_weights(self.training_model.get_weights())
            # reset the update counter
            self.update_counter = 0
            print('done!')

        return loss

    def train_on_model(self, dummy_targets, masks, state_batch, targets):
        loss = self.training_model.train_on_batch([state_batch, targets, masks], [dummy_targets, targets])
        #loss = self.training_model.train_on_batch([state_batch], [targets])
        return loss

    def predict_on_target(self, next_state_batch):
        target_q_values = self.target_model.predict_on_batch(next_state_batch)
        q_batch = np.max(target_q_values, axis=1).flatten()
        return q_batch

    def decrease_explore_rate(self):
        # Linear annealed: f(x) = ax + b.
        a = -float(self.epsilon_max - self.epsilon_min) / float(EPSILON_DECAY_RATE)
        b = float(self.epsilon_max)
        value = a * float(self.current_step) + b
        self.epsilon = max(self.epsilon_min, value)

    def load(self, name):
        self.run_model.load_weights(name)

    def save(self, name):
        self.training_model.save_weights(name, overwrite=True)


def train(warmup_steps=500):


    """
     The training process will create a state containing 4 images in grayscale from the observation.
     For each step that is taken a next_state is created, which will contain images from the previous three steps
     and the new observation in grayscale.

     To start out the first state will be based on the observation from env.reset and copied three times.
    """

    highscore = 0

    step = 0
    loss = 0
    save_counter = 0

    while step < STEPS:

        agent.current_step = step

        # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
        state = process_observation(
            env.reset()
        )

        # reshape the state to fit input to the convolutional network. The dimensions must be 4 x 84 x 84.
        # That is 4 images in grayscale with 84 x 84 pixels
        input_state = build_state(
            state,
            state,
            state,
            state
        )

        score = 0
        start_over = True

        while True:

            if not agent.training:
                env.render()

            # env.render()

            if start_over:
                # start the game
                action = env.action_space.sample()
                start_over = False
            else:
                # Predict an action to take based on the 4 images that represents the current state
                action = agent.act(input_state)

            next_state, reward, is_done, info = env.step(action)
            score += reward
            reward = np.clip(reward, -1, 1)

            if agent.save_state:
                save_state(action, input_state, step)

            # Get the first observation and make it grayscale and reshape to 84 x 84 pixels
            next_state = process_observation(next_state)

            # reshape the state to fit input to the convolutional network. The dimensions must be 4 x 84 x 84.
            # That is 4 images in grayscale with 84 x 84 pixels
            # Swap out the first image and replace with the former next_state
            input_next_state = build_state(
                input_state[0][1],
                input_state[0][2],
                input_state[0][3],
                next_state
            )

            if agent.training:
                # Add the beginning state of this action and the outcome to the replay memory
                # if reward != 0.0:
                agent.remember(input_state, action, reward, input_next_state, is_done)

            # Move forward...
            input_state = input_next_state
            step += 1
            save_counter += 1

            # If the game has stopped, sum up the result and continue to next episode
            if is_done:
                if score > highscore:
                    highscore = score

                print('')
                print("step: {}/{}, score: {}, highscore: {}, steps: {}, e: {}, loss: {}"
                      .format(step, STEPS, score, highscore, step, agent.epsilon, loss[0] if isinstance(loss, list) else loss))
                break

            # If we have remembered observations that exceeds the batch_size (32), we should replay them.
            if agent.training and step > warmup_steps and step % TRAIN_INTERVAL == 0:
                loss = agent.replay()

        if agent.training:
            agent.decrease_explore_rate()
            if save_counter > SAVE_RATE:
                print('Saving model....')
                agent.save("../save/breakout-dqn-v2.h5")
                save_counter = 0
                print('done!')

    if agent.training:
        agent.replay()
        print('Saving final model....')
        agent.save("../save/breakout-dqn-v2.h5")
        print('done training!')


def save_state(action, input_state, step):
    save_image(input_state[0][0], "img_1.png", step)
    save_image(input_state[0][1], "img_2.png", step)
    save_image(input_state[0][2], "img_3.png", step)
    save_image(input_state[0][3], "img_4.png", step)
    save_action(action, step)


def build_state(image_1, image_2, image_3, image_4):
    return reshape_to_fit_network(
        np.array(
            (
                image_1,
                image_2,
                image_3,
                image_4
            )
        )
    )


def reshape_to_fit_network(input):
    return input.reshape(1, input.shape[0], input.shape[1], input.shape[2])  # 1*84*84*4


def save_image(img_array, file_name, folder_name):
    folders = "images/{}".format(folder_name)

    if not os.path.exists(folders):
        os.makedirs(folders)

    img = Image.fromarray(img_array)
    img.save("{}/{}".format(folders, file_name))


def save_action(action, folder_name):
    folders = "images/{}".format(folder_name)

    if not os.path.exists(folders):
        os.makedirs(folders)

    with open("{}/{}".format(folders, "action.txt"), "w") as file:
        file.write(str(action))


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

    action_size = env.action_space.n
    print('action size')
    print(action_size)

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())

    logger = tf.summary.FileWriter('./logs')

    model = DQNModel(state_size=(WINDOW_LENGTH,) + INPUT_SHAPE, action_size=action_size, logger=logger)

    agent = DQNAgent(model)

    if args['mode'] == 'run':
        agent.epsilon = 0.05
        agent.training = False
        agent.load("../save/breakout-dqn-v2.h5")



    #if args['save_state']:
    #    agent.save_state = True

    train()
