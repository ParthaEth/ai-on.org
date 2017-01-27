# Description

Often we want to have feed back loops in neural networks. This was the original motivation behind the development of recurrent neural networks. As a result we have the simple RNN and the LSTM and many more kinds of RNNs. Though a standard implementation of such networks as layers lets us stack them and build a deeper(in space) RNN they do not let us feed back a computation across different layers. This project is to develop a Keras module that will let one design flexibly feed back connections just like feed forward connections.

# Detailed description

Consider the following architecture.
![img_20170125_201108](https://cloud.githubusercontent.com/assets/10944728/22304296/1c197ad4-e337-11e6-80a6-1734175f7c04.png) 
it is really really hard to code in keras. Now you would say why not the following code
```python
from keras.layers import Dense, Input
from keras.models import Model

input = Input(batch_shape=(2, 5), dtype='float32', name='current_pose_input')

d1 = Dense(5)
d2 = Dense(5)
d3 = Dense(5)

o = d1(input)
o = d2(o)
o = d3(o)
o = d1(o)

d = Dense(5)(o)

m = Model(input=input, output=d)
m.compile('rmsprop', 'mse')

from keras.utils.visualize_util import plot
plot(m, to_file = 'exampleRec.png')
```
but htis isn't really the architecture you think you've coded. The network is just as follows.
![img_20170125_201109](https://cloud.githubusercontent.com/assets/10944728/22305102/8c85c6d0-e33a-11e6-81d1-a9c5d8792ff4.jpg)

So basically it is just one time step unrolling. Also if you have an input dim as `(n * batch_size, time_steps, data_dim)` then you will have to flatten them(in time) so basically all hell breaks loose and you have to take care of everything manually. Further more there is the following problem. Consider the following image

![img_20170125_201110](https://cloud.githubusercontent.com/assets/10944728/22328399/512c66ac-e3bc-11e6-8431-87e3483177d8.jpg)

if you have taken the pain to read the image, you can see that what we really want is the one shown above above but in general we would get the below one if you do it naively like the code I showed above, which is like stacking RNN layes. 

Finally this is not really so hard to code I believe. It is like flattening everything inside the feedback loop and we have to take special care of the recurrent layers inside and also keep in mind the statefull-ness. This is nasty to do in the user code and is the only thing missing in the keras framework.

# Why is it important?
Lastly the implementation will allow users of kers to design any kind of RNN structure as easily as they could code NNs that are directed acyclic graphs (i.e. no feed back or stateless). With heightened focus towards RNNs and deep RNNs people are likely to try to code novel architectures like skip connections in the feed back connections and stuff like that. As a motivation try coding this network and you will see the problems of performing a time rollout manyally in the user side code
![img_20170125_201111](https://cloud.githubusercontent.com/assets/10944728/22291461/d192eb7e-e306-11e6-85ef-4e1e497e8da8.jpg)
if you are wondering where this architecture came from please read [this paper](https://arxiv.org/pdf/1506.02216v6.pdf)

# Help + Who might be interested
I @parthaEth am interested and would be happy to team up. I might already go forward with it all by my self but it will be great to have a team. Please have a look at [this issue](https://github.com/fchollet/keras/issues/5160) to understand it perhaps a bit more clearly.
