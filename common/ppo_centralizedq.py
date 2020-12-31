import tensorflow as tf
import numpy as np

class ValueNetwork():
    def __init__(self, num_features, hidden_size, num_output, learning_rate=.01):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_output = num_output
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.get_variable("W3", shape=[self.hidden_size, self.num_output])
            ]
            self.B = [
                tf.get_variable("B1", [self.hidden_size]),
                tf.get_variable("B2", [self.hidden_size]),
                tf.get_variable("B3", [self.num_output]),
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]) + self.B[1])
            self.output = tf.matmul(self.layer_2, self.W[2]) + self.B[2]

            self.target = tf.placeholder(shape=[None], dtype=tf.float32)
            self.activated_action = tf.placeholder(shape=[None, self.num_output], dtype=tf.float32)

            self.loss = tf.multiply(self.output, self.activated_action)
            self.loss = tf.reduce_sum(self.loss, axis=-1)
            self.loss = tf.losses.mean_squared_error(self.loss, self.target)
            self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.session.run(init)

    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    def update(self, states, actions, discounted_rewards):
        _, loss = self.session.run([self.minimize, self.loss], feed_dict={
            self.observations: states,
            self.activated_action: actions,
            self.target: discounted_rewards
        })

class ValueNetwork2D():
    def __init__(self, num_features, hidden_size, num_output1D, num_output2D, learning_rate=.01):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_output1D = num_output1D
        self.num_output2D = num_output2D
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.get_variable("W3", shape=[self.hidden_size, self.num_output1D * self.num_output2D])
            ]
            self.B = [
                tf.get_variable("B1", [self.hidden_size]),
                tf.get_variable("B2", [self.hidden_size]),
                tf.get_variable("B3", [self.num_output1D * self.num_output2D]),
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]) + self.B[1])
            self.output = tf.matmul(self.layer_2, self.W[2]) + self.B[2]
            self.output = tf.reshape(self.output, shape=[-1, self.num_output1D, self.num_output2D])

            self.target = tf.placeholder(shape=[None, self.num_output2D], dtype=tf.float32)
            self.activated_action = tf.placeholder(shape=[None, self.num_output1D, self.num_output2D], dtype=tf.float32)

            self.loss = tf.multiply(self.output, self.activated_action)
            self.loss = tf.reduce_sum(self.loss, axis=1)
            self.loss = tf.losses.mean_squared_error(self.loss, self.target)
            self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.session.run(init)

    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    def update(self, states, actions, discounted_rewards):
        _, loss = self.session.run([self.minimize, self.loss], feed_dict={
            self.observations: states,
            self.activated_action: actions,
            self.target: discounted_rewards
        })
