import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')



    return fig


class train_param:
    def __init__(self):
        self.n_epoch = 10
        self.batch_size = 64
        self.z_dim = 100


train_param_ = train_param()

class fc:
    def __init__(self,out_dim,weights = None,bias = None, activation = None, scope = ""):
        self.out_dim = out_dim
        self.weights = weights
        self.bias = bias
        self.activation = activation   
        self.scope = scope
    
    def build(self,input_):
        batch,in_dim = input_.get_shape()
        
        if self.weights:
            assert self.weights.get_shape() == (in_dim,self.out_dim)
        else:
            shape_ = (in_dim,self.out_dim)
            self.weights = tf.get_variable('weigths',shape = shape_,initializer= tf.contrib.layers.xavier_initializer(),trainable=True)
        
        if self.bias:
            assert self.bias.get_shape() == (self.out_dim,)
        else:
            shape_ = (self.out_dim,)
            
            self.bias = tf.get_variable('bias',shape = shape_,initializer= tf.contrib.layers.xavier_initializer(),trainable=True)
        
        if self.activation:
            return self.activation(tf.matmul(input_,self.weights)+self.bias)
        
        return tf.matmul(input_,self.weights)+self.bias
    
    def call(self,input_):
        if self.scope:
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE) as scope:
                return self.build(input_)
        
        else:
            return self.build(input_)


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])


def discriminator(x):
    layer_1 = fc(128,activation=tf.nn.relu, scope='D_1')
    fc1 = layer_1.call(x)
    logits = fc(1,scope = 'D_2').call(fc1)
    prob = tf.nn.sigmoid(logits)

    return prob

def generator(z):
    fc1 = fc(392,activation= tf.nn.relu,scope = 'G_1').call(z)
    fc2 = fc(784,activation= tf.nn.sigmoid, scope = 'G_2').call(fc1)

    return fc2 


x = tf.placeholder(tf.float32,shape = [None,784])
z = tf.placeholder(tf.float32,shape = [None,train_param_.z_dim])


G_sample = generator(z)
real = discriminator(x)
fake = discriminator(generator(z))


theta_G = [v for v in tf.trainable_variables() if 'G' in v.name]
theta_D = [v for v in tf.trainable_variables() if 'D' in v.name]




d_loss = -tf.reduce_mean(tf.log(real)+tf.log(1.0-fake))
g_loss = -tf.reduce_mean(tf.log(fake))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer(0.0001).minimize(d_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

for it in range(20000):
    X_mb, _ = mnist.train.next_batch(train_param_.batch_size)

    _, D_loss_curr = sess.run([D_solver, d_loss], feed_dict={x: X_mb, z: sample_Z(train_param_.batch_size,train_param_.z_dim)})
    _, G_loss_curr = sess.run([G_solver, g_loss], feed_dict={z: sample_Z(train_param_.batch_size,train_param_.z_dim)})

    if it % 1000 == 0:
        print(it)
        print('d:',D_loss_curr)
        print('g:',G_loss_curr)

    if it % 10000 == 0:
        i = 0
        samples = sess.run(G_sample, feed_dict={z: sample_Z(train_param_.batch_size, train_param_.z_dim)})
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)