from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as f
import tensorflow as tf

CSV_COLUMN_NAMES = ["SepalLenght","SepalWidth","PetalLenght","PetalWidth","Species"]
SPECIES = ["Setosa","Versicolor","Virginica"]

train_path = tf.keras.utils.get_file(
    "iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop("Species")
test_y = test.pop("Species")
#Input function
def input_fn(features, labels, training=True, batch_size=256):
    #Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    #Shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30,10],
    n_classes=3)
classifier.train(
    input_fn = lambda: input_fn(train,train_y, training=True),steps=5000)
eval_result = classifier.evaluate( input_fn = lambda: input_fn(test,test_y, training=False))
print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

def input_fn (feature, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ["SepalLenght","SpealWidth","PetalLength","PetalWidth"]
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit():valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    print(pred_dict)
    class_id = pred_dict["class_ids"][0]
    probability = pred_dict["probabilities"][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100*probability))
