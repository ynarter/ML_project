import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        self.mean = np.mean(training_data, axis=0)

        centered_data = training_data - self.mean

        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        self.W = sorted_eigenvectors[:, :self.d]
        
        explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        exvar = np.sum(explained_variance[:self.d]) * 100
        
        return exvar


    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        centered_data = data - self.mean
        W = self.W  
        data_reduced = centered_data @ W
        
        print("Shape of data before reduction:", data.shape)
        print("Shape of data after reduction:", data_reduced.shape)
        
        return data_reduced
    
    def reconstruct(self, data):
        """
        Reconstruction of the data using the compressed representation found previously
        Arguments:
            data (array): data of shape (N,D)
        Returns:
            recovered_data (array): recovered data of shape (N,D)
        
        """
        recovered_data = self.mean + self.W @ self.reduce_dimension(data)
        
        return recovered_data

