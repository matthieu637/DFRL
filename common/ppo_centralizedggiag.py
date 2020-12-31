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

            self.rollout = tf.placeholder(shape=[None, self.num_output], dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(self.output, self.rollout)
            self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.session.run(init)

    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    def update(self, states, discounted_rewards):
        _, loss = self.session.run([self.minimize, self.loss], feed_dict={
            self.observations: states, self.rollout: discounted_rewards
        })


class PPOPolicyNetwork():
    def __init__(self, num_features, layer_size, num_actions, num_agent, epsilon=.1,
                 learning_rate=9e-4):
        self.tf_graph = tf.Graph()
        self.num_agent = num_agent

        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[num_features, layer_size]),
                tf.get_variable("W2", shape=[layer_size, layer_size]),
                tf.get_variable("W3", shape=[layer_size, num_actions ** num_agent])
            ]
            self.B = [
                tf.get_variable("B1", [layer_size]),
                tf.get_variable("B2", [layer_size]),
                tf.get_variable("B3", [num_actions ** num_agent]),
            ]
            trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.saver = tf.train.Saver(trainable_vars, max_to_keep=3000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]) + self.B[1])
            self.output = tf.matmul(self.output, self.W[2]) + self.B[2]
            self.output = tf.nn.softmax(self.output)

            self.advantages = [tf.placeholder(shape=[None], dtype=tf.float32) for _ in range(num_agent)]

            self.chosen_actions = tf.placeholder(shape=[None, num_actions ** num_agent], dtype=tf.float32)
            self.old_probabilities = tf.placeholder(shape=[None, num_actions ** num_agent], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            self.ratio = self.new_responsible_outputs / self.old_responsible_outputs

            self.loss = [tf.minimum(
                tf.multiply(self.ratio, self.advantages[i]),
                tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages[i])) for i in range(num_agent)]
            self.loss = [self.loss[i] - 0.03 * self.new_responsible_outputs * tf.log(self.new_responsible_outputs + 1e-10) for i in range(num_agent)]
            self.loss = [-tf.reduce_mean(self.loss[i], axis=0) for i in range(num_agent)]

            self.W0_grad = tf.placeholder(dtype=tf.float32)
            self.W1_grad = tf.placeholder(dtype=tf.float32)
            self.W2_grad = tf.placeholder(dtype=tf.float32)

            self.B0_grad = tf.placeholder(dtype=tf.float32)
            self.B1_grad = tf.placeholder(dtype=tf.float32)
            self.B2_grad = tf.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad, self.B0_grad, self.B1_grad,
                                          self.B2_grad]
            self.trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.gradients = [[(np.zeros(var.get_shape()), var) for var in self.trainable_vars] for i in range(num_agent)]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = [self.optimizer.compute_gradients(self.loss[i], self.trainable_vars) for i in range(num_agent)]
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.global_variables_initializer()
            self.session.run(init)

    def get_dist(self, states):
        dist = self.session.run(self.output, feed_dict={self.observations: states})
        return dist

    def update(self, states, chosen_actions, ep_advantages, sorted_omega):
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})

        grad = np.array(self.gradients)
        grad = np.tensordot(sorted_omega, grad, axes=(0, 0))
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: grad[0][0],
            self.W1_grad: grad[1][0],
            self.W2_grad: grad[2][0],
            self.B0_grad: grad[3][0],
            self.B1_grad: grad[4][0],
            self.B2_grad: grad[5][0],
        })

        fd = {self.observations: states,
              self.chosen_actions: chosen_actions,
              self.old_probabilities: old_probabilities}
        for i in range(self.num_agent):
            fd[self.advantages[i]]=ep_advantages[:, i]
        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict=fd)

    def save_w(self, name):
        self.saver.save(self.session, name + '.ckpt')

    def restore_w(self, name):
        self.saver.restore(self.session, name + '.ckpt')
