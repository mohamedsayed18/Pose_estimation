import cv2
import numpy as np
import os
import csv


def load_image(image):
    """reshape and convert image to fit the model"""
    img = cv2.imread(image)     # Load images
    img = cv2.resize(img, (257, 257), interpolation=cv2.INTER_LINEAR)  #  resize
    img = (np.float32(img) - 127.5) / 127.5   # change image to float and normalize
    img = img.reshape((1, 257, 257, 3))    # resize
    return img


def get_vectors(folder_path, interpreter):
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
        data = np.append(kp.reshape(1,34), 1)
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
