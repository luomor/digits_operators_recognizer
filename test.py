# My modules

from ocr import image
from computing import compute

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy

def get_model(box_size, numb_classes):

    acc = Accuracy()
    network = input_data(shape=[None, box_size, box_size, 1])

    # Conv layers
    network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_1')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_2')

    network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_3')
    network = max_pool_2d(network, 2, strides=2)

    # Fully Connected Layer
    network = fully_connected(network, 1024, activation='tanh')
    # Dropout layer
    network = dropout(network, 0.5)
    # Fully Connected Layer
    network = fully_connected(network, numb_classes, activation='softmax')
    # Final network
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001, metric=acc)

    # The model with details on where to save
    # Will save in current directory
    model = tflearn.DNN(network, checkpoint_path='model-', best_checkpoint_path='best-model-', tensorboard_verbose=0)

    return model

import os
import numpy as np

# with open(os.path.join('ai', 'classes.txt'), 'r') as desc:
#     # Split string read on whitespace
#     classes = desc.read().split()
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def predict(pattern, model, box_size=32):

    # Reshape from flattened to box
    pattern = pattern.reshape((box_size, box_size, 1))

    # Predict probabilities
    one_hot = model.predict([pattern])[0] # Predict
    # Get index of the highest probability
    index = np.argmax(one_hot)

    return classes[index]

# Restore model
model_abs_path = os.path.join('.', 'ai', 'model', 'mathocr.model')

model = get_model(box_size=32, numb_classes=10)
# Restore model's pre-trained parameters
model.load(model_abs_path)

image_abs_path="./002.jpeg";
# Extract patterns
patterns = image.extract_patterns(image_abs_path)

# Run recognizer on each separate pattern
prediction = ''
for pattern in patterns:
    prediction += predict(pattern, model)
    print('prediction:', prediction)
results = compute.get_all(prediction)
print(results)
