# Description

Often we want to have feed back loops in neural networks. This was the original motivation behind the development of recurrent neural networks. As aresult we have the simple RNN and the LSTM and many more kinds of RNNs. Though a standard implementation of such networks as a layer lets us stack them and build a deeper(in space) RNN they do not let us feed back a computation accross different layers. This project is to develope a Keras module that will let one design flexibly feed back connections just like feed forward connections.

# Detailed description

Consider the following architecture.
