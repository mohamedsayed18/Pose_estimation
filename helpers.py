import cv2
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim


def load_image(image):
    """reshape and convert image to fit the model"""
    img = cv2.imread(image)     # Load images
    img = cv2.resize(img, (257, 257), interpolation=cv2.INTER_LINEAR)  #  resize
    img = (np.float32(img) - 127.5) / 127.5   # change image to float and normalize
    img = img.reshape((1, 257, 257, 3))    # resize
    return img


def get_vectors(folder_path, label, interpreter):
    """"""
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()  # the input shape
    output_details = interpreter.get_output_details()  # the output shape

    for i in os.listdir(folder_path):
        image = load_image(folder_path + '/' + i)  # load image


        # Feed image to the model
        interpreter.set_tensor(input_details[0]['index'], image)  # feed the image to model
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])  # the heatmap output
        output_offset = interpreter.get_tensor(output_details[1]['index'])  # the offset output
        heatmap = output_data[0]
        offsets = output_offset[0]

        con, kp = get_keypoints(heatmap, offsets)
        data = np.append(kp.reshape(1,34), label)
        # Write the keypoint in CSV file
        with open('poses.csv', mode='a') as pose_file:
            employee_writer = csv.writer(pose_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow(data)

def get_keypoints(heatmap, offsets):
  """
  Get the keypoint output
  params:
  heatmap: the heatmap of the output
  offsets: the offset vectors of the output
  return:
  keypointPositions: list of the (x, y) position of every keypoint
  """
  # store some variables
  confidences =[]
  offset_vectors=[]
  hm_positions = []

  scores = sigmoid(heatmap)   # sigmoid
  for k in range(17):   # no. of keypoints
    x,y = np.unravel_index(np.argmax(scores[:,:,k]), scores[:,:,k].shape)   # find the max
    hm_positions.append([x,y])
    confidences.append(scores[x,y,k])
    offset_vectors.append([offsets[x,y,k], offsets[x,y,k+17]])
  keypointPositions = np.add(np.array(hm_positions) * 32, offset_vectors)

  return confidences, keypointPositions


def sigmoid(a):
  """sigmoid function"""
  return 1/(1 + np.exp(-a))

# Build new Network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.l1 = nn.Linear(34, 20)
        self.l2 = nn.Linear(20, 10)
        self.l3 = nn.Linear(10, 3)
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.sig(self.l1(x))
        x = self.sig(self.l2(x))
        x = self.sig(self.l3(x))
        return x

def train(net, data_set):
    criterion = nn.CrossEntropyLoss()   # cross entropy
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # optimizer
    vector, label = data_set[:, :-1], data_set[:, -1]
    for i in range(data_set.shape[0]):
        inputs, labels = torch.Tensor(vector[i].reshape(1,34)), torch.tensor([np.long(label[i])])
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return