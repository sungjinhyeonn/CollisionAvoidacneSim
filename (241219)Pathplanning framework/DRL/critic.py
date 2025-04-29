import tensorflow as tf
import numpy as np
import math
import os

# 하이퍼파라미터
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 0.0001
TAU = 0.001

class Critic:
    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Critic 네트워크 생성
        self.state_input, self.action_input, self.q_value_output, self.net = self.create_network()
        
        # Target 네트워크 생성
        self.target_state_input, self.target_action_input, self.target_q_value_output, self.target_update = self.create_target_network()
        
        # 학습 방법 정의
        self.create_training_method()
        
        # 모델 저장 경로
        self.model_dir = os.path.join(os.path.dirname(__file__), 'model', 'critic')
        os.makedirs(self.model_dir, exist_ok=True)

    def create_network(self):
        state_input = tf.compat.v1.placeholder(tf.float32, [None, self.state_dim])
        action_input = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim])
        
        # 첫 번째 레이어 (상태)
        W1 = self.variable([self.state_dim, LAYER1_SIZE], self.state_dim)
        b1 = self.variable([LAYER1_SIZE], self.state_dim)
        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        
        # 두 번째 레이어 (상태 + 행동)
        W2 = self.variable([LAYER1_SIZE, LAYER2_SIZE], LAYER1_SIZE + self.action_dim)
        W2_action = self.variable([self.action_dim, LAYER2_SIZE], LAYER1_SIZE + self.action_dim)
        b2 = self.variable([LAYER2_SIZE], LAYER1_SIZE + self.action_dim)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2)
        
        # 출력 레이어 (Q-값)
        W3 = tf.Variable(tf.random.uniform([LAYER2_SIZE, 1], -0.003, 0.003))
        b3 = tf.Variable(tf.random.uniform([1], -0.003, 0.003))
        q_value_output = tf.matmul(layer2, W3) + b3
        
        return state_input, action_input, q_value_output, [W1, b1, W2, W2_action, b2, W3, b3]

    def create_target_network(self):
        state_input = tf.compat.v1.placeholder(tf.float32, [None, self.state_dim])
        action_input = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim])
        
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(self.net)
        target_net = [ema.average(x) for x in self.net]
        
        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        q_value_output = tf.matmul(layer2, target_net[5]) + target_net[6]
        
        return state_input, action_input, q_value_output, target_update

    def create_training_method(self):
        self.y_input = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def train(self, y_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def predict_target(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def predict(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def update_target(self):
        self.sess.run(self.target_update)

    def variable(self, shape, f):
        return tf.Variable(tf.random.uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))

    def load_network(self):
        self.saver = tf.compat.v1.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded critic network")
        else:
            print("Could not find old network weights")

    def save_network(self, time_step):
        print('save critic-network...', time_step)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'critic-network'), global_step=time_step)