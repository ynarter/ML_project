import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=3000, task_kind="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None   
        self.task_kind = task_kind 

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of sshape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###s
        #### WRITE YOUR CODE HERE!
        ###
        ##
        print_period = 1000
        training_labels = label_to_onehot(training_labels)
        D = training_data.shape[1]  # number of features
        C = training_labels.shape[1]  # number of classes
        # Random initialization of the weights
        self.weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = gradient_logistic_multi(training_data, training_labels, self.weights)
            self.weights = self.weights - self.lr * gradient

            predictions = self.predict(training_data)
            if accuracy_fn(predictions, onehot_to_label(training_labels)) == 100:
                break
            #logging and plotting
            if print_period and it % print_period == 0:
                print('loss at iteration', it, ":", loss_logistic_multi(training_data, training_labels, self.weights))
                print(f"acc: {accuracy_fn(predictions, onehot_to_label(training_labels))}")
                
        return predictions

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        probabilities = f_softmax(test_data, self.weights)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

def loss_logistic_multi(data, labels, w):
    """ 
    Loss function for multi class logistic regression, i.e., multi-class entropy.
    
    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        w (array): Weights of shape (D, C)
    Returns:
        float: Loss value 
    """
    N = data.shape[0]
    probabilities = f_softmax(data, w)
    cross_entropy = -np.sum(labels * np.log(probabilities))/N
    return cross_entropy

def gradient_logistic_multi(data, labels, W):
    """
    Compute the gradient of the entropy for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        W (array): Weights of shape (D, C)
    Returns:
        grad (np.array): Gradients of shape (D, C)
    """
    probabilities = f_softmax(data, W)
    error = probabilities - labels
    grad = np.dot(data.T, error)
    return grad

def accuracy_fn(labels_pred, labels_gt):
    """
    Computes the accuracy of the predictions.
    
    Args:
        labels_pred (array): Predicted labels of shape (N,)
        labels_gt (array): GT labels of shape (N,)
    Returns:
        acc (float): Accuracy, in range [0, 100].
    """
    return np.sum(labels_pred == labels_gt) / labels_gt.shape[0] * 100.0

def f_softmax(data, W):
    """
    Softmax function for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        W (array): Weights of shape (D, C) where C is the number of classes
    Returns:
        array of shape (N, C): Probability array where each value is in the
            range [0, 1] and each row sums to 1.
            The row i corresponds to the prediction of the ith data sample, and 
            the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
    """
    z = np.dot(data, W)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)