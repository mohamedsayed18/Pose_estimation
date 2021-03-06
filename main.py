import tensorflow as tf
from numpy import loadtxt
from helpers import *
"""
model = tf.lite.Interpreter("posenet.tflite")   # load the model

# Get the vectors out of the images
get_vectors('Data_set/Prayer_pose', 0, model)
get_vectors('Data_set/Straw_pose', 1, model)
get_vectors('Data_set/X_pose', 2, model)
"""
# feed vectors to the classifier
mynet = MyNetwork()     # create model
data = loadtxt('poses.csv', delimiter=',', dtype=np.float)      # data
train(mynet, 5, data)  # train the

# save
torch.save(mynet, "./torch_model_v1.pt")

# load

