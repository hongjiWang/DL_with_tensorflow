""" DNNRegression with coustom input_fn for housing dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age",
            "dis", "tax", "ptratio"]
LABEL = "medv"

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

def main(unused_argv):
    # load datasets
    training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

    # set of 6 examples for which to predict the median housing values
    prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    # feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    # Build 2 layer fully connected DNN with 10, 10 units respectively
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                              hidden_units=[10, 10],
                              model_dir="/tmp/boston_model")
    # train
    regressor.train(input_fn=get_input_fn(training_set), steps=5000)

    # evaluate loss over one epoch of test_set
    ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    loss_score = ev['loss']
    print('Loss: {0:f}'.format(loss_score))

    # print out predictions over a slice of precdition_set
    y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1,
                                            shuffle=False))
    # .predict() return a iterator of dics, convert it to a list and
    # print predictions
    predictions = list(p['predictions'] for p in itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
    tf.app.run()

