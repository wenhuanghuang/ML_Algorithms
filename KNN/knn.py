import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.features = None
        self.labels = None

    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels = labels

    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        :param point: List[float]
        :return:  List[int]
        """
        # rank features based on distance
        ranker = list()
        # calculate distance to the point
        for index, feature in enumerate(self.features):
            dist = self.distance_function(feature, point)
            ranker.append([dist, index])
        # sort based on distance, and prioritize small index if there is a tie
        ranker = sorted(ranker, key=lambda x:(x[0], x[1]))
        # k nearest neighbours
        neighbours = ranker[:self.k]
        neighbours = np.array(neighbours).T[1].astype(int)
        return neighbours.tolist()

    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        # the predictions are based on majority of votes of knn
        result = list()
        for index, feature in enumerate(features):
            k_neighbors = self.get_k_neighbors(feature)
            k_labels = [self.labels[x] for x in k_neighbors]
            major_vote = sum(k_labels) > (self.k // 2)
            result.append(major_vote)
        return result

if __name__ == '__main__':
    print(np.__version__)
