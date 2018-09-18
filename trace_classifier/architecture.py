from keras.layers import Input, Dense, Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Model


def architecture(n_classes=3, input_shape=(15, 2)):
    """
    Model architecture, modified from https://arxiv.org/abs/1510.03820v4.

    Input must be named `input`, and output layer must be named `output`, otherwise
    inference script fail to find the input and output tensor from a graph.

    Parameters
    ----------
    n_classes: Integer.
               Number of classes.
    input_shape: Tuple of integers.
                 Shape of the input data (excluding batch size).

    Returns
    -------
    A Keras model.
    """

    n1 = 16
    n2 = 12

    inputs = Input(shape=input_shape, name='input')

    x = inputs

    x0 = Conv1D(n1, kernel_size=7, strides=1, activation='relu')(x)
    x0 = GlobalMaxPooling1D()(x0)

    x1 = Conv1D(n1, kernel_size=8, strides=1, activation='relu')(x)
    x1 = GlobalMaxPooling1D()(x1)

    x = concatenate([x0, x1], axis=1)
    x = Dropout(0.2)(x)
    x = Dense(n2)(x)
    x = Activation('selu')(x)
    x = Dropout(0.2)(x)

    x = Dense(n_classes, activation='softmax', name='output')(x)
    outputs = x

    return Model(inputs=inputs, outputs=outputs)
