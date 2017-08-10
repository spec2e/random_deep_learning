# -*- coding: utf-8 -*-
import random
import argparse
from collections import deque

import gym
import numpy as np

np.random.seed(1337)  # for reproducibility

import os
import keras
import keras.backend as K
import tensorflow as tf

from PIL import Image
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Convolution2D, Activation, Flatten, Permute
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

STEPS = 1000000
EPSILON_DECAY_RATE = 500000

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

TARGET_MODEL_UPDATE_RATE = 2500
SAVE_RATE = 10000

LOG_INTERVAL = 1000

BATCH_SIZE = 128
TRAIN_INTERVAL = 8


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
        self.model = self.build_training_model()
        self.run_model = self.build_run_model()
        self.target_model = self.build_target_model()
        self.update_counter = 0
        self.training = True
        self.tbCallBack = TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        _model = Sequential()
        _model.add(Permute((2, 3, 1), input_shape=state_size))
        _model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, 4)))
        _model.add(Activation('relu'))
        _model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        _model.add(Activation('relu'))
        _model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        _model.add(Activation('relu'))
        _model.add(Flatten())
        _model.add(Dense(512))
        _model.add(Activation('relu'))
        _model.add(Dense(self.action_size))
        _model.add(Activation('linear'))

        return _model

    def build_target_model(self):

        _model = self._build_model()

        _model.compile(optimizer='sgd', loss='mse')

        return _model

    def build_run_model(self):

        _model = self._build_model()
        _optimizer = Adam(lr=.00025)
        _model.compile(optimizer=_optimizer, loss='mse')

        return _model


    def build_training_model(self):

        _model = self._build_model()

        def Model(input, output, **kwargs):
            if int(keras.__version__.split('.')[0]) >= 2:
                return keras.models.Model(inputs=input, outputs=output, **kwargs)
            else:
                return keras.models.Model(input=input, output=output, **kwargs)


        def huber_loss(y_true, y_pred, clip_value):
            # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
            # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
            # for details.
            x = y_true - y_pred
            condition = K.abs(x) < clip_value
            squared_loss = .5 * K.square(x)
            linear_loss = clip_value * (K.abs(x) - .5 * clip_value)

            if hasattr(tf, 'select'):
                return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
            else:
                return tf.where(condition, squared_loss, linear_loss)  # condition, true, false

        _delta_clip = 1.
        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, _delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        def mean_q(y_true, y_pred):
            return K.mean(K.max(y_pred, axis=-1))

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = _model.output
        y_true = Input(name='y_true', shape=(self.action_size,))
        mask = Input(name='mask', shape=(self.action_size,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])

        trainable_model = Model(input=[_model.input, y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2

        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred)  # we only include this for the metrics
        ]

        _optimizer = Adam(lr=.00025)

        trainable_model.compile(optimizer=_optimizer, loss=losses, metrics=[mean_q])

        return trainable_model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

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

        target_q_values = self.target_model.predict_on_batch(next_state_batch)
        q_batch = np.max(target_q_values, axis=1).flatten()

        # Discount the Q value from the network. Gamma is the discount factor
        discounted_reward_batch = self.gamma * q_batch

        # Set discounted reward to zero for all states that were terminal.
        # If we died in this state, make sure it is zero!
        discounted_reward_batch *= terminal1_batch

        targets = np.zeros((BATCH_SIZE, self.action_size))
        dummy_targets = np.zeros((BATCH_SIZE,))
        masks = np.zeros((BATCH_SIZE, self.action_size))

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
        loss = self.model.train_on_batch([state_batch, targets, masks], [dummy_targets, targets])

        if self.update_counter > TARGET_MODEL_UPDATE_RATE:
            print('setting weights on target_model...')
            self.target_model.set_weights(self.model.get_weights())
            # reset the update counter
            self.update_counter = 0
            print('done!')

        return loss

    def decrease_explore_rate(self):
        # Linear annealed: f(x) = ax + b.
        a = -float(self.epsilon_max - self.epsilon_min) / float(EPSILON_DECAY_RATE)
        b = float(self.epsilon_max)
        value = a * float(self.current_episode) + b
        self.epsilon = max(self.epsilon_min, value)

    def load(self, name):
        self.run_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name, overwrite=True)


def train(warmup_steps=5000):


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
        agent.current_episode = step

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

            # save_image(input_state[0][0], "img_1.png", step)
            # save_image(input_state[0][1], "img_2.png", step)
            # save_image(input_state[0][2], "img_3.png", step)
            # save_image(input_state[0][3], "img_4.png", step)
            # save_action(action, step)

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
                      .format(step, STEPS, score, highscore, step, agent.epsilon, loss))
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
    state_size = (WINDOW_LENGTH,) + INPUT_SHAPE
    print('state_size: ', state_size)
    action_size = env.action_space.n
    print('action size')
    print(action_size)

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())

    agent = DQNAgent(state_size, action_size)

    if args['mode'] == 'run':
        agent.epsilon = 0.05
        agent.training = False
        agent.load("../save/breakout-dqn-v2.h5")


    train()
