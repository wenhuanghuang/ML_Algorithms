import numpy as np
from knn import KNN

# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    assert len(real_labels) == len(predicted_labels)
    real = np.array(real_labels)
    pred = np.array(predicted_labels) 
    numerator = sum(real * pred)
    denominator = sum(real) + sum(pred)
    if denominator == 0:
        return 0
    f1_score = 2 * numerator * 1.0 / denominator
    return f1_score

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        p1 = np.array(point1)
        p2 = np.array(point2)
        return sum((abs(p1 - p2))**3)**(1 / 3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        
        p1 = np.array(point1)
        p2 = np.array(point2)
        return sum((p1 - p2)**2)**(1 / 2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        
        p1 = np.array(point1)
        p2 = np.array(point2)
        deno = (sum(p1**2)**(1/2) * sum(p2**2)**(1/2))
        if deno == 0:
            return 1
        cos_simi_dist = 1 - sum(p1 * p2) / deno 
        return cos_simi_dist

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        self.best_f1 = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        f1_results = list()
        models = list()
        # loop over different distance functions
        for func_key in distance_funcs:
            dist_func = distance_funcs[func_key]
            f1_result_for_func = []
            # loop over different k values
            for k in range (1, 30, 2):
                # train the model
                knn_classifier = KNN(k, dist_func)
                knn_classifier.train(x_train, y_train)
                # prediction on validation set
                y_pred = knn_classifier.predict(x_val)
                # calculate f1 on this k and save it to the list
                f1 = f1_score(y_val, y_pred)
                f1_result_for_func.append(f1)
                # save the model to the list
                models.append(knn_classifier)
            # f1 for each distance function
            f1_results.append(f1_result_for_func)

        f1_array = np.array(f1_results)
        # get index of best result (highest f1)
        model_index = np.argmax(f1_array)
        # get index in 2d
        best_f1 = float(np.max(f1_array))
        ind = np.unravel_index(model_index, f1_array.shape)
        function_list = [x for x in distance_funcs.keys()]

        # You need to assign the final values to these variables
        self.best_k = ind[1] * 2 + 1
        self.best_distance_function = function_list[ind[0]]
        self.best_model = models[model_index]
        self.best_f1 = best_f1

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        f1_results = list()
        models = list()
        # loop over each scaling scheme
        for scaler_key in scaling_classes:
            f1_result_for_scaler = []
            scaler = scaling_classes[scaler_key]()
            # loop over each distance function
            for func_key in distance_funcs:
                dist_func = distance_funcs[func_key]
                f1_result_for_func = []
                # loop over each k value
                for k in range (1, 30, 2):
                    # train the model
                    knn_classifier = KNN(k, dist_func)
                    # apply scaling in training set
                    knn_classifier.train(scaler(x_train), y_train)
                    # apply scaling in validation set and make predictions
                    y_pred = knn_classifier.predict(scaler(x_val))
                    # calculate f1 score
                    f1 = f1_score(y_val, y_pred)
                    f1_result_for_func.append(f1)
                    models.append(knn_classifier)
                f1_result_for_scaler.append(f1_result_for_func)
            f1_results.append(f1_result_for_scaler)
            
        f1_array = np.array(f1_results)
        best_f1 = float(np.max(f1_array))
        # get index of best model
        model_index = np.argmax(f1_array)
        # get index in 3d
        # ind[0] for scaler, ind[1] for distance function, ind[2] for k value
        ind = np.unravel_index(model_index, f1_array.shape)
        
        function_list = [x for x in distance_funcs]
        scaler_list = [x for x in scaling_classes]
        # You need to assign the final values to these variables
        self.best_k = ind[2] * 2 + 1
        self.best_distance_function = function_list[ind[1]]
        self.best_scaler = scaler_list[ind[0]]
        self.best_model = models[model_index]
        self.best_f1 = best_f1

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        
        features = np.array(features)
        norm = np.linalg.norm(features, axis=1).reshape((-1,1))
        norm[norm == 0] = 1
        result = features / norm
        return result.tolist()

class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        mins = np.min(features, axis=0).reshape((1,-1))
        maxs = np.max(features, axis=0).reshape((1,-1))

        match = maxs == mins
        mins[match] = 0
        maxs[match] = 1
        result = (features - mins)/(maxs - mins)
        return result.tolist()