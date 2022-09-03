
import tensorflow as tf
from tf_agents.networks import sequential


class ModelWrapper:
    def __init__(self, num_actions, fc_layer_params) -> None:
        self.fc_layer_params = fc_layer_params
        self.num_actions = num_actions
        self.model = None

        self._build()

    @staticmethod
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, 
                mode='fan_in', 
                distribution='truncated_normal'
            )
        )

    def _build(self):
        
        flatten_layer = tf.keras.layers.Flatten()
        dense_layers = [self.dense_layer(num_units) for num_units in self.fc_layer_params]

        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

        self.model = sequential.Sequential([flatten_layer] + dense_layers + [q_values_layer])


if __name__ == '__main__':
    fc_layer_params = [10, 15, 20]
    num_actions = 6

    wrapper = ModelWrapper(num_actions, fc_layer_params)

    inputs = tf.ones([3, 20, 14, 3])

    pred = wrapper.model(inputs)

    b = 1
