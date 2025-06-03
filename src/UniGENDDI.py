import os
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import csv
import pickle

import graph
import keras
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.utils import to_categorical

event_num = 2
droprate = 0.4
vector_size = 128
SPLIT = "../split/demo"


def DNN():
    train_input1 = Input(shape=(456,), name="Inputlayer1")
    train_input2 = Input(shape=(456,), name="Inputlayer2")
    # train_input1 = Input(shape=(vector_size * 5 + 456,), name="Inputlayer1")
    # train_input2 = Input(shape=(vector_size * 5 + 456,), name="Inputlayer2")

    # Attention Neural Network
    train_input = keras.layers.Concatenate()([train_input1, train_input2])

    attention_probs = Dense(
        456,
        activation="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name="attention1",
    )(train_input)

    att = Dense(
        456,
        activation="softmax",
        kernel_initializer="random_uniform",
        name="attention",
    )(attention_probs)

    vec = keras.layers.Multiply()([train_input1, train_input2])
    attention_mul = keras.layers.Multiply()([vec, att])

    # Deep neural network classifier
    train_in = Dense(4096, activation="relu", name="FullyConnectLayer1")(attention_mul)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(2048, activation="relu", name="FullyConnectLayer2")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(1024, activation="relu", name="FullyConnectLayer3")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(512, activation="relu", name="FullyConnectLayer4")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(256, activation="relu", name="FullyConnectLayer5")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(128, activation="relu", name="FullyConnectLayer6")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(64, activation="relu", name="FullyConnectLayer7")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(event_num, name="SoftmaxLayer")(train_in)

    out = Activation("softmax", name="OutputLayer")(train_in)

    model = Model(inputs=[train_input1, train_input2], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def calculate_metric_score(real_labels, predict_score):
    # Evaluate the prediction performance
    precision, recall, pr_thresholds = precision_recall_curve(
        real_labels, predict_score
    )
    aupr_score = auc(recall, precision)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + recall[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    print("all_F_measure: ")
    print(all_F_measure)
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
    auc_score = auc(fpr, tpr)

    f = f1_score(real_labels, predict_score)
    print("F_measure:" + str(all_F_measure[max_index]))
    print("f-score:" + str(f))
    accuracy = accuracy_score(real_labels, predict_score)
    precision = precision_score(real_labels, predict_score)
    recall = recall_score(real_labels, predict_score)
    print("results for feature:" + "weighted_scoring")
    print(
        "*** AUC score:%.3f, AUPR score:%.3f, precision score:%.3f, recall score:%.3f, f score:%.3f,accuracy:%.3f ***"
        % (auc_score, aupr_score, precision, recall, f, accuracy)
    )
    results = {
        "auc_score": auc_score,
        "aupr_score": aupr_score,
        "precision": precision,
        "recall": recall,
        "f": f,
        "accuracy": accuracy,
    }

    return results


def cross_validation():
    #  Build the drug-drug interaction network
    g = graph.Graph()
    g.read_edgelist("input/ogbl_dkp/mapping/drug_drug.txt")
    print(g.G.number_of_edges())

    # Remove the test_links in the network
    edge_list = {}
    for split in ["train", "val", "test"]:
        edge_list[split] = pickle.load(open(f"{SPLIT}/{split}.pkl", "rb"))

    for split in ["val", "test"]:
        edges = edge_list[split]["edge"]
        for i in range(edges.shape[0]):
            n1, n2 = edges[0, i], edges[1, i]
            if g.G.has_edge(str(n1), str(n2)):
                g.G.remove_edge(str(n1), str(n2))

    
    print("Test Begin")

    model_n2v = loadmodel("../embeddings/n2v.csv")
    model_sage = loadmodel("../embeddings/sage_emb.csv")

    # Concatenate of representation vectors 
    E = np.zeros((841, 456), float)
    for i in range(841):
        E[i][0:200] = model_n2v[i]
        E[i][200:456] = model_sage[i]

    # Training set
    X_train1 = []
    X_train2 = []
    Y_train = []

    train_pos = edge_list["train"]["edge"]
    label_pos = [1] * train_pos.shape[1]
    pos_x1 = E[train_pos[0, :]]
    pos_x2 = E[train_pos[1, :]]
    Y_train.extend(label_pos)

    train_neg = edge_list["train"]["edge_neg"]
    label_neg = [0] * train_neg.shape[1]
    neg_x1 = E[train_neg[0, :]]
    neg_x2 = E[train_neg[1, :]]
    Y_train.extend(label_neg)

    X_train1 = np.concatenate([pos_x1, neg_x1], axis=0)
    X_train2 = np.concatenate([pos_x2, neg_x2], axis=0)
    Y_train = np.array(Y_train)
    Y_train = to_categorical(Y_train, 2)
    del train_pos, train_neg, label_pos, label_neg

    print(X_train1.shape, X_train2.shape, Y_train.shape)
    dnn = DNN()
    dnn.fit(
        [X_train1, X_train2],
        Y_train,
        batch_size=128,
        epochs=150,
        shuffle=True,
        verbose=1,
    )

    # Test set
    X_test1 = []
    X_test2 = []
    Y_test = []

    test_pos = edge_list["test"]["edge"]
    label_pos = [1] * test_pos.shape[1]
    pos_x1 = E[test_pos[0, :]]
    pos_x2 = E[test_pos[1, :]]
    Y_test.extend(label_pos)

    test_neg = edge_list["test"]["edge_neg"]
    label_neg = [0] * test_neg.shape[1]
    neg_x1 = E[test_neg[0, :]]
    neg_x2 = E[test_neg[1, :]]
    Y_test.extend(label_neg)

    X_test1 = np.concatenate([pos_x1, neg_x1], axis=0)
    X_test2 = np.concatenate([pos_x2, neg_x2], axis=0)
    del test_pos, test_neg, label_pos, label_neg

    y_pred_label = dnn.predict([X_test1, X_test2])
    y_pred_label = np.argmax(y_pred_label, axis=1)
    y_pred_label = np.array(y_pred_label).tolist()
    print("Final Predicted Labels:", y_pred_label)

    return calculate_metric_score(Y_test, y_pred_label)


def loadmodel(path):
    # Load the files of representation vectors generated before
    Emb = pd.read_csv(path, header=None)
    features = list(Emb.columns)
    Emb = np.array(Emb[features])
    return Emb


def main(runs):
    log = open("demo_result.txt", "w")
    results = []
    for run in range(runs):
        print("Executing Run", run)
        result = cross_validation()
        log.write(json.dumps(result) + "\n")
        results.append(result)

    # find the mean and std
    auc_score = np.array([result["auc_score"] for result in results])
    aupr_score = np.array([result["aupr_score"] for result in results])
    precision = np.array([result["precision"] for result in results])
    recall = np.array([result["recall"] for result in results])
    f = np.array([result["f"] for result in results])
    accuracy = np.array([result["accuracy"] for result in results])
    final_results = {
        "auc_score": str(np.mean(auc_score)) + "+/-" + str(np.std(auc_score)),
        "aupr_score": str(np.mean(aupr_score)) + "+/-" + str(np.std(aupr_score)),
        "precision": str(np.mean(precision)) + "+/-" + str(np.std(precision)),
        "recall": str(np.mean(recall)) + "+/-" + str(np.std(recall)),
        "f": str(np.mean(f)) + "+/-" + str(np.std(f)),
        "accuracy": str(np.mean(accuracy)) + "+/-" + str(np.std(accuracy)),
    }
    log.write(json.dumps(final_results) + "\n")
    log.close()


if __name__ == "__main__":
    main(5)
