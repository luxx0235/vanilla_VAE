import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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


def encoder(input_):
    fc1 = fc(392,activation= tf.nn.relu,scope='fc1').call(input_)
    fc2 = fc(196,activation= tf.nn.relu,scope='fc2').call(fc1)
    fc3 = fc(98,activation= tf.nn.relu,scope='fc3').call(fc2)

    mu = fc(train_param_.z_dim,scope='encode_mu').call(fc3)
    sigma = fc(train_param_.z_dim,scope='encode_sigma').call(fc3)

    return (mu,sigma)


def sample_z(args):
    mu,sigma = args
    eps = tf.random.normal(shape = (train_param_.batch_size,train_param_.z_dim))

    return mu + tf.exp(sigma/2)*eps


def decoder(input_):
    fc1 = fc(98,activation= tf.nn.relu,scope='fc4').call(input_)
    fc2 = fc(196, activation= tf.nn.relu,scope='fc5').call(fc1)
    fc3 = fc(392,activation= tf.nn.relu,scope='fc6').call(fc2)

    out = fc(784,scope = 'fc7').call(fc3)

    return out



x = tf.placeholder(tf.float32,shape = [None,784])
z = tf.placeholder(tf.float32, shape=[None, train_param_.z_dim])


encoded = encoder(x)
mu,sigma = encoded
sampled_z = sample_z(encoded)
decoded = decoder(sampled_z)
X_samples = decoder(z)




#recon_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x,decoded), 1)
#recon_loss = tf.reduce_mean(tf.squared_difference(decoded,x))
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded, labels=x), 1)
KL = 0.5*tf.reduce_sum(tf.exp(sigma)+tf.math.square(mu)- 1.0-sigma, 1)

vae_loss = tf.reduce_mean(recon_loss + KL)



"""
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded, labels=x), 1)
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(sigma) + mu**2 - 1. - sigma, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)
"""

solver = tf.train.AdamOptimizer(0.001).minimize(vae_loss)


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(20000):
    X_mb, _ = mnist.train.next_batch(train_param_.batch_size)

    _, loss = sess.run([solver,vae_loss], feed_dict={x: X_mb})
    print(it,':',loss)

"""
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, train_param_.z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
"""
