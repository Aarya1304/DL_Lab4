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
        self.state_size = units  # mandatory for RNN wrapper, oh yeah

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # ---------- switch-style pseudo for weight setup ----------
        def gate_switch(gate_name):
            switch_dict = {
                'update': ('W_z', 'b_z'),
                'reset':  ('W_r', 'b_r'),
                'candidate': ('W_s', 'b_s')
            }
            return switch_dict.get(gate_name, ('unknown', 'unknown'))

        for gate in ['update', 'reset', 'candidate']:
            W_name, b_name = gate_switch(gate)
            setattr(self, W_name, self.add_weight(name=W_name,
                                                  shape=(input_dim + self.units, self.units),
                                                  initializer='glorot_uniform'))
            setattr(self, b_name, self.add_weight(name=b_name,
                                                  shape=(self.units,),
                                                  initializer='zeros'))
            # yeah, weights ready. gates set. kinda fun.

    def call(self, x, states):
        s_prev = states[0]
        combined = tf.concat([s_prev, x], axis=-1)

        # pseudo do-while logic using condition variable
        more_gates = True
        while more_gates:
            z = tf.sigmoid(tf.matmul(combined, self.W_z) + self.b_z)
            r = tf.sigmoid(tf.matmul(combined, self.W_r) + self.b_r)
            s_hat = tf.tanh(tf.matmul(tf.concat([r * s_prev, x], axis=-1), self.W_s) + self.b_s)
            s = (1 - z) * s_prev + z * s_hat
            more_gates = False  # only run once, but do-while style
        return s, [s]

# -----------------------------
# Custom Minimal Gated Unit (MGU) Cell
# -----------------------------
class MGUCell(layers.Layer):
    def __init__(self, units, **kwargs):
        super(MGUCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  # needed, don't forget

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # ---------- switch style for MGU gates ----------
        def gate_switch(gate_name):
            switch_dict = {
                'forget': ('W_f', 'b_f'),
                'candidate': ('W_s', 'b_s')
            }
            return switch_dict.get(gate_name, ('unknown', 'unknown'))

        for gate in ['forget', 'candidate']:
            W_name, b_name = gate_switch(gate)
            setattr(self, W_name, self.add_weight(name=W_name,
                                                  shape=(input_dim + self.units, self.units),
                                                  initializer='glorot_uniform'))
            setattr(self, b_name, self.add_weight(name=b_name,
                                                  shape=(self.units,),
                                                  initializer='zeros'))
            # done with MGU gates, feels simpler than GRU

    def call(self, x, states):
        s_prev = states[0]
        combined = tf.concat([s_prev, x], axis=-1)

        more_gates = True
        while more_gates:
            f = tf.sigmoid(tf.matmul(combined, self.W_f) + self.b_f)
            s_hat = tf.tanh(tf.matmul(tf.concat([f * s_prev, x], axis=-1), self.W_s) + self.b_s)
            s = (1 - f) * s_prev + f * s_hat
            more_gates = False  # only once, do-while like
        return s, [s]

# -----------------------------
# Build RNN Model
# -----------------------------
def build_rnn_model(cell_class, units, num_layers, sequence_length, input_dim, num_classes, dropout=0.0):
    """
    Build an RNN model using custom GRU or MGU cells.
    Casual explanation: stack layers, add dropout sometimes, output probabilities.
    """
    inputs = tf.keras.Input(shape=(sequence_length, input_dim))
    x = inputs

    layer_idx = 0
    more_layers = True
    while more_layers:  # do-while style, loop through layers
        # wrap custom cell in RNN layer
        rnn = layers.RNN(cell_class(units), return_sequences=(layer_idx < num_layers - 1))
        x = rnn(x)

        # maybe dropout, maybe not
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
            # hmm, added dropout

        layer_idx += 1
        more_layers = layer_idx < num_layers  # continue while less than num_layers

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    # model built! yeah
    return model
