import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Convolution2D, Activation, Flatten, Permute
from keras.optimizers import Adam

import keras.backend as K
import tensorflow as tf

class DQNModel:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        _model = Sequential()
        _model.add(Permute((2, 3, 1), input_shape=self.state_size))
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

        def clipped_masked_error(args):
            y_true, y_pred, mask = args

            # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
            # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
            # for details.
            x = y_true - y_pred
            _delta_clip = 1.
            condition = K.abs(x) < _delta_clip
            squared_loss = .5 * K.square(x)
            linear_loss = _delta_clip * (K.abs(x) - .5 * _delta_clip)

            if hasattr(tf, 'select'):
                loss = tf.select(condition, squared_loss, linear_loss)  # condition, true, false
            else:
                loss = tf.where(condition, squared_loss, linear_loss)  # condition, true, false

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

        # append the y_true and the mask input to the last Dense layer (output layer)
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])

        trainable_model = Model(input=[_model.input, y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        print(trainable_model.summary())
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred)  # we only include this for the metrics
        ]

        _optimizer = Adam(lr=.00025)

        trainable_model.compile(optimizer=_optimizer, loss=losses, metrics=[mean_q])

        return trainable_model
