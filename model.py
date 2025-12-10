# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------
# Custom GRU Cell
# -----------------------------
class GRUCell(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GRUCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  # required for RNN wrapper

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Update gate
        self.W_z = self.add_weight(name='W_z', shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform')
        self.b_z = self.add_weight(name='b_z', shape=(self.units,), initializer='zeros')
        # Reset gate
        self.W_r = self.add_weight(name='W_r', shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform')
        self.b_r = self.add_weight(name='b_r', shape=(self.units,), initializer='zeros')
        # Candidate state
        self.W_s = self.add_weight(name='W_s', shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform')
        self.b_s = self.add_weight(name='b_s', shape=(self.units,), initializer='zeros')

    def call(self, x, states):
        s_prev = states[0]
        combined = tf.concat([s_prev, x], axis=-1)

        z = tf.sigmoid(tf.matmul(combined, self.W_z) + self.b_z)
        r = tf.sigmoid(tf.matmul(combined, self.W_r) + self.b_r)
        s_hat = tf.tanh(tf.matmul(tf.concat([r * s_prev, x], axis=-1), self.W_s) + self.b_s)
        s = (1 - z) * s_prev + z * s_hat
        return s, [s]

# -----------------------------
# Custom Minimal Gated Unit (MGU) Cell
# -----------------------------
class MGUCell(layers.Layer):
    def __init__(self, units, **kwargs):
        super(MGUCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  # required for RNN wrapper

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Forget gate
        self.W_f = self.add_weight(name='W_f', shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform')
        self.b_f = self.add_weight(name='b_f', shape=(self.units,), initializer='zeros')
        # Candidate state
        self.W_s = self.add_weight(name='W_s', shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform')
        self.b_s = self.add_weight(name='b_s', shape=(self.units,), initializer='zeros')

    def call(self, x, states):
        s_prev = states[0]
        combined = tf.concat([s_prev, x], axis=-1)

        f = tf.sigmoid(tf.matmul(combined, self.W_f) + self.b_f)
        s_hat = tf.tanh(tf.matmul(tf.concat([f * s_prev, x], axis=-1), self.W_s) + self.b_s)
        s = (1 - f) * s_prev + f * s_hat
        return s, [s]

# -----------------------------
# Build RNN Model
# -----------------------------
def build_rnn_model(cell_class, units, num_layers, sequence_length, input_dim, num_classes, dropout=0.0):
    """
    Build an RNN model using custom GRU or MGU cells.
    """
    inputs = tf.keras.Input(shape=(sequence_length, input_dim))
    x = inputs

    for layer_idx in range(num_layers):
        # Wrap custom cell with RNN layer
        rnn = layers.RNN(cell_class(units), return_sequences=(layer_idx < num_layers - 1))
        x = rnn(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
