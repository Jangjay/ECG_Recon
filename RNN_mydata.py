from matplotlib import pyplot as plt
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
import math
import time
import progressbar

#
# save_file = './lead1.ckpt'
###########DATA LOAD#####################

ECG_3lead_trainingset=np.load('D:/F드라이브/논문데이터 정리/WindowingDATA/파이썬 트레이닝데이터/trainingset250/3lead_resamp.npy')

ECG_12lead_lead1_trainingset=np.load('D:/F드라이브/논문데이터 정리/WindowingDATA/파이썬 트레이닝데이터/trainingset250/lead1_resamp.npy')


Data_index = np.load('D:/F드라이브/논문데이터 정리/WindowingDATA/파이썬 트레이닝데이터/Data_index.npy')
# np.save('C:/Users/Sohn Jang Jay/Desktop/Data_index.npy', Data_index)
ECG_3lead_trainingset_Shuffle=np.copy(ECG_3lead_trainingset)
ECG_12lead_lead1_trainingset_Shuffle=np.copy(ECG_12lead_lead1_trainingset)

ECG_3lead_trainingset_Shuffle=ECG_3lead_trainingset_Shuffle[Data_index,:,:]
ECG_12lead_lead1_trainingset_Shuffle=ECG_12lead_lead1_trainingset_Shuffle[Data_index,:]

ECG_12lead_lead1_trainingset_Shuffle=ECG_12lead_lead1_trainingset_Shuffle[:,:, np.newaxis]


X_DATA=np.load('D:/F드라이브/논문데이터 정리/WindowingDATA/파이썬 테스트데이터/testX/subject1(testX).npy')
Y_DATA=np.load('D:/F드라이브/논문데이터 정리/WindowingDATA/파이썬 테스트데이터/testY/lead1/subject1(testX).npy')


ECG_3lead_testset=X_DATA.copy()
ECG_12lead_lead1_testset=Y_DATA.copy()

test_x=[];
test_x=ECG_3lead_testset[np.newaxis,0:250,:]
test_y=[];
test_y=ECG_12lead_lead1_testset[0:250].copy()
test_y=test_y[np.newaxis,0:250,np.newaxis]


# plt.plot(test_x[0,:,:],label=r"one beat")
#
# plt.legend(loc="lower left", fontsize=14)
# plt.xlabel("Time")
# plt.ylabel("Value", rotation=0)
# plt.show()
#
# print(ECG_3lead_trainingset_Shuffle.shape)
# print(ECG_12lead_lead1_trainingset_Shuffle.shape)
#
# plt.figure(figsize=(11,4))
# plt.title("ECG_Lead1", fontsize=14)
# plt.plot(ECG_12lead_lead1_trainingset_Shuffle[5000,:,0],label=r"one beat")
# plt.legend(loc="lower left", fontsize=14)
# plt.xlabel("Time")
# plt.ylabel("Value", rotation=0)
# plt.show()
#
################
# Layer Params #
################
n_steps = 250
n_neurons = 100
n_inputs = 3
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])


# RNN Model using OutputProjectionWrapper
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs)
predictions, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

################
# Train Params #
################
learning_rate = 0.001
n_iterations = 1000000
batch_size = 1000
epochnumber=100000
# loss
mse = tf.losses.mean_squared_error(labels=y, predictions=predictions)
# optimizer
# train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(mse)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)
iternumber=int(160483/batch_size)

saver = tf.train.Saver()
# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    temp = []
    for epoch in range(epochnumber):
        Train_Data_index = np.random.permutation(len(ECG_3lead_trainingset_Shuffle))
        for iteration in range(iternumber):
            batch_x = []
            batch_y = []
            batch_x = np.copy(ECG_3lead_trainingset_Shuffle[Train_Data_index[iteration*batch_size:(iteration+1)*batch_size-1], :, :])
            batch_y = np.copy(ECG_12lead_lead1_trainingset_Shuffle[Train_Data_index[iteration*batch_size:(iteration+1)*batch_size-1], :,:])
            sess.run(train_op, feed_dict={X: batch_x, y: batch_y})


        if epoch % 1 == 0:
            loss = mse.eval(feed_dict={X: batch_x, y: batch_y})
            temp.append(loss)
            print('Epoch: {:06d}, MSE: {:.4f}'.format(epoch, loss))


    saver.save(sess, "./lead1_model")

#
# print('y_pred:{}\n{}'.format(y_pred.shape, y_pred))
#
# plt.title("Testing the Model", fontsize=14)
# plt.plot(test_y[0,:,0], "w", markersize=10, label="target", color='yellow')
# plt.plot(y_pred[0,:,0], "r", markersize=10, label="prediction")
# plt.legend(loc="upper left")
# plt.xlabel("Time")
# plt.show()
