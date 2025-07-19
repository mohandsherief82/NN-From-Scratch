import numpy as np
import seaborn as sns # Still assuming for visualization, not core NN logic

class NeuralNetwork:

    def __init__(self, X, Y, hidden_layer_nodes: list = None):
        """
        Initializes the Neural Network.

        Args:
            X (np.array): Input features, shape (n_features, n_samples).
            Y (np.array): True labels, shape (n_samples,).
            hidden_layer_nodes (list, optional): A list specifying the number of
                                                  nodes in each hidden layer.
        """
        self.X = X
        self.n_features, self.m_samples = self.X.shape
        self.Y = Y

        num_classes = len(np.unique(self.Y))

        # Construct layer_dims: [input_features, hidden_layer_nodes..., output_nodes]
        if hidden_layer_nodes is None:

            # Default to one hidden layer if not specified
            self.layer_dims = [self.n_features, num_classes, num_classes]

            # More reasonable default if no_layers was a param: [self.n_features, N_HIDDEN, num_classes]
            # For simplicity, if no hidden_layer_nodes, let's just make it a 1-hidden-layer network
            # with 100 nodes if not specified
            self.layer_dims = [self.n_features, 100, num_classes] # Default 100 nodes for one hidden layer
        else:
            self.layer_dims = [self.n_features] + hidden_layer_nodes + [num_classes]

        self.no_layers = len(self.layer_dims) - 1 # Number of weight/bias pairs (connections between layers)

        # Initializing parameters dynamically
        self.W = {}
        self.b = {}

        # Caches for forward propagation
        self.Z = {}
        self.A = {"Layer 0": self.X}

        # Caches for backward propagation gradients
        self.dZ = {}
        self.dW = {}
        self.db = {}

        for layer in range(1, self.no_layers + 1):
            # Weights: (nodes in current layer, nodes in previous layer)
            self.W[f"Layer {layer}"] = np.random.rand(self.layer_dims[layer], self.layer_dims[layer - 1]) - 0.5

            # Biases: (nodes in current layer, 1)
            self.b[f"Layer {layer}"] = np.random.rand(self.layer_dims[layer], 1) - 0.5

            # Initialize dW, db, dZ for all layers for consistency
            self.dW[f"Layer {layer}"] = np.zeros_like(self.W[f"Layer {layer}"])
            self.db[f"Layer {layer}"] = np.zeros_like(self.b[f"Layer {layer}"])
            self.dZ[f"Layer {layer}"] = np.zeros((self.layer_dims[layer], self.m_samples)) # Initialize with correct shape


    @staticmethod
    def _ReLU(Z):
        return np.maximum(Z, 0)


    @staticmethod
    def _softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # Subtract max for numerical stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


    def _forward_prop(self):
        for layer in range(1, self.no_layers + 1):
            # A from previous layer is Z[f"Layer {layer - 1}"] from your current setup.
            # This is slightly confusing. Usually Z stores pre-activation, A stores post-activation.
            # If Z["Layer 0"] is X, then A["Layer 0"] should be X too.
            # Let's adjust to use A for previous layer's activation.
            A_prev = self.A[f"Layer {layer - 1}"]


            self.Z[f"Layer {layer}"] = self.W[f"Layer {layer}"].dot(A_prev) + self.b[f"Layer {layer}"]


            if layer == self.no_layers:
                self.A[f"Layer {layer}"] = self._softmax(self.Z[f"Layer {layer}"])
            else:
                self.A[f"Layer {layer}"] = self._ReLU(self.Z[f"Layer {layer}"])


    @staticmethod
    def _ReLU_deriv(Z):
        return (Z > 0).astype(float) # Explicitly cast to float


    def _one_hot(self):
        num_classes = self.layer_dims[-1] # Get num_classes from the last layer in layer_dims
        one_hot_Y = np.zeros((num_classes, self.Y.size))
        one_hot_Y[self.Y, np.arange(self.Y.size)] = 1
        return one_hot_Y


    def _backward_prop(self):
        Y_one_hot = self._one_hot()


        # Iterate backwards through layers
        # The loop runs from the last layer (self.no_layers) down to 1 (the first hidden layer).
        for layer in range(self.no_layers, 0, -1): # Corrected range to include layer 1


            if layer == self.no_layers:
                self.dZ[f"Layer {layer}"] = self.A[f"Layer {layer}"] - Y_one_hot
            else:
                # For hidden layers, dZ[l] = W[l+1].T * dZ[l+1] * g'(Z[l])
                self.dZ[f"Layer {layer}"] = self.W[f"Layer {layer + 1}"].T.dot(self.dZ[f"Layer {layer + 1}"]) * self._ReLU_deriv(self.Z[f"Layer {layer}"])


            # dW[l] = dZ[l] . A[l-1].T / m
            # db[l] = sum(dZ[l]) / m
            self.dW[f"Layer {layer}"] = self.dZ[f"Layer {layer}"].dot(self.A[f"Layer {layer - 1}"].T) / self.m_samples # Use A for previous layer
            self.db[f"Layer {layer}"] = np.sum(self.dZ[f"Layer {layer}"], axis=1, keepdims=True) / self.m_samples # Sum across samples for bias


    def _update_params(self, alpha):
        for layer in range(1, self.no_layers + 1):
            self.W[f"Layer {layer}"] = self.W[f"Layer {layer}"] - alpha * self.dW[f"Layer {layer}"]
            self.b[f"Layer {layer}"] = self.b[f"Layer {layer}"] - alpha * self.db[f"Layer {layer}"]


    def get_predictions(self):
        return np.argmax(self.A[f"Layer {self.no_layers}"], 0)


    def get_accuracy(self, predictions):
        return np.sum(predictions == self.Y) / self.Y.size


    def gradient_descent(self, iterations, alpha):
        for i in range(1, iterations + 1):
            self._forward_prop()
            self._backward_prop()
            self._update_params(alpha)
            if i % 10 == 0 or i == iterations or i == 1:
                print("Iteration: ", i)
                print(f"Accuracy: {float(np.floor((self.get_accuracy(self.get_predictions()) * 10000)) / 100)}%")

