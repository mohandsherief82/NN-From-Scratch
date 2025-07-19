import numpy as np
import pandas as pd
from helpers import NeuralNetwork

def main():
    data = pd.read_csv("MNSIT/train.csv")

    data = np.array(data)
    m,n = data.shape # get number of row "m" and columns "n"

    # randomizing data before splitting
    np.random.shuffle(data)

    # getting the training dataset ready for use and applying .T for transpose
    data = data[1:m].T
    Y_train = data[0] # final value
    X_train = data[1:n] # pixel value
    X_train = X_train / 255.

    # Training the neural network
    neuron = NeuralNetwork(X=X_train, Y=Y_train)
    print()
    W = neuron.gradient_descent(iterations= 250, alpha=0.2)


if __name__ == "__main__":
    main()
