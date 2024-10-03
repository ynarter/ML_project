import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind #either "classification" or "regression"
        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = self.predict(training_data)
        
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        
        test_labels_list = []
        for sample in test_data:
            distances = np.linalg.norm(self.training_data - sample, axis=1) #distances between the sample and all training data
            nearest_indices = np.argsort(distances)[:self.k] #indices of the k nearest neighbors
            nearest_labels = self.training_labels[nearest_indices] #labels of the k nearest neighbors
            
            if self.task_kind == "classification":
                #predict the label with majority vote for classification task 
                predicted_label = np.argmax(np.bincount(nearest_labels))

            elif self.task_kind == "regression":
                #for regression task we can use the average values
                predicted_label = np.mean(nearest_labels, axis=0)
            
            test_labels_list.append(predicted_label)
            
        test_labels = np.array(test_labels_list)
        return test_labels

