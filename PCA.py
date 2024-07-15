import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X_scaled):
        """
        Fits the PCA model to the given scaled data.

        Parameters:
        X_scaled (ndarray): The scaled data.

        Returns:
        None
        """
        # Calculate the covariance matrix
        covariance_matrix = np.cov(X_scaled)

        # Calculate the eigen values and eigen vectors
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

        # Sort the eigen values in descending order and select the top n_components
        sorted_key = np.argsort(eigen_values)[::-1][: self.n_components]

        # Store the sorted eigen values and eigen vectors
        self.eigen_values = eigen_values[sorted_key]
        self.eigen_vectors = eigen_vectors[:, sorted_key]

    def transform(self, X):
        """
        Transforms the input data by projecting it onto the eigenvectors.

        Parameters:
        X (ndarray): The data to be transformed.

        Returns:
        ndarray: The transformed data projected onto the eigenvectors.
        """
        # Calculate the transformation of the input data by multiplying it with the eigenvectors.
        # np.dot() performs a matrix multiplication between X and self.eigen_vectors.
        # The result is a matrix where each row is the projection of the corresponding row of X onto the eigenvectors.
        # This transformation reduces the dimensionality of the data to the number of eigenvectors used for PCA.
        # The result is a representation of the data in a lower dimensional space called principal components.
        transformed_data = np.dot(X, self.eigen_vectors)

        return transformed_data

    def fit_transform(self, X):
        """
        Fits the PCA model to the given scaled data and transforms it.

        Parameters:
        X (ndarray): The scaled data.

        Returns:
        ndarray: The transformed data projected onto the eigenvectors.

        This function first calls the fit() function to calculate the eigen values and eigen vectors
        based on the scaled data. Then it calls the transform() function to project the scaled data
        onto the eigenvectors and returns the transformed data.
        """
        # Fit the PCA model to the scaled data
        self.fit(X)

        # Transform the scaled data by projecting it onto the eigenvectors
        transformed_data = self.transform(X)

        return transformed_data
