from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import *
import numpy as np
from matplotlib import pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():
    batch_1 = unpickle("data_batch_1")
    batch_2 = unpickle("data_batch_2")
    batch_3 = unpickle("data_batch_3")
    batch_4 = unpickle("data_batch_4")
    batch_5 = unpickle("data_batch_5")
    testImages = unpickle("test_batch")
    scale(batch_1[b"data"])
    scale(batch_2[b"data"])
    scale(batch_3[b"data"])
    scale(batch_4[b"data"])
    scale(batch_5[b"data"])
    scale(testImages[b"data"])
    classifier = MLPClassifier((2058,1029,515,),batch_size = 512)
    classifier.fit(batch_1[b"data"], batch_1[b"labels"])
    classifier.fit(batch_2[b"data"], batch_2[b"labels"])
    classifier.fit(batch_3[b"data"], batch_3[b"labels"])
    classifier.fit(batch_4[b"data"], batch_4[b"labels"])
    classifier.fit(batch_5[b"data"], batch_5[b"labels"])
    y_test = testImages[b"labels"]
    predictions = classifier.predict(testImages[b"data"])
    confusionMatrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    numErrors = 10000 - sum(np.diag(confusionMatrix))
    print("Numero di iterazioni:",classifier.n_iter_)
    print("Numero di errori:",numErrors)
    print("Accuratezza:", accuracy)
    fig, ax = plt.subplots()
    ax.plot(range(len(classifier.loss_curve_)), classifier.loss_curve_)
    ax.set_title("Loss function")
    ax.set_ylabel("loss")
    ax.set_xlabel("iteration")
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title("Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=range(10),
                                            display_labels=["airplane", "automobile", "bird", "cat", "deer", "dog",
                                                            "frog", "horse", "ship", "truck"], ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
