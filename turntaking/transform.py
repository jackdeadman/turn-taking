from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg


@dataclass
class FeatureTransform(ABC):
    num_features: int
    std_scaler: StandardScaler = field(default_factory=StandardScaler)

    def fit(self, data):
        data = self.std_scaler.fit_transform(data)
        self._fit(data)

    def transform(self, data):
        data = self.std_scaler.transform(data)
        return self._transform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def _fit(self, data):
        pass

    @abstractmethod
    def _transform(self, data):
        pass


@dataclass
class PCATransform_v2(FeatureTransform):

    eigenvectors: np.array = None
    eigenvalues: np.array = None

    def expected_points(self, data):
        return self.inverse_transform(self.transform(data))

    def inverse_transform(self, data):
        """
        :param data: (samples, self.num_features)
        :return:
        """

        # (N, num_features)
        projection_matrix = self.eigenvectors[:, :self.num_features]

        # (samples, N)
        original_dim_space = np.dot(data, projection_matrix.T)

        original_space = self.std_scaler.inverse_transform(original_dim_space)
        return original_space

    def _fit(self, data):
        # Data already normalised
        cov = np.cov(data, rowvar=False)
        self.eigenvalues, self.eigenvectors = scipy.linalg.eigh(cov)
        self.eigenvectors = self.eigenvectors[::-1]
        self.eigenvalues = self.eigenvalues[::-1]
        assert np.alltrue(np.isreal(self.eigenvectors)), 'Complex eigenvectors computed'

    def _transform(self, data):
        """
        :param data: (samples, features)
        :return:
        """
        projection_matrix = self.eigenvectors[:, :self.num_features]
        projected = np.dot(data, projection_matrix)
        return projected


@dataclass
class PCATransform(FeatureTransform):

    __model: PCA = None

    def __post_init__(self):
        self.__model = PCA(n_components=self.num_features)

    def _fit(self, data):
        self.__model.fit(data)

    def _transform(self, data):
        return self.__model.transform(data)

    @property
    def eigenvalues(self):
        return self.__model.explained_variance_

    @property
    def eigenvectors(self):
        return self.__model.components_


@dataclass
class TSNETransform(FeatureTransform):
    perplexity: float = 40
    n_iter: int = 10000
    random_state: int = None

    __model: TSNE = None

    def __post_init__(self):
        self.__model = TSNE(
            n_components=self.num_features,
            n_iter=self.n_iter, perplexity=self.perplexity,
            random_state=self.random_state
        )

    def _fit(self, data):
        self.__model.fit(data)

    def _transform(self, data):
        return self.__model.fit_transform(data)

