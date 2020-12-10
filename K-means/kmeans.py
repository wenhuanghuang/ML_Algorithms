import numpy as np

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''
    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # implement the Kmeans++ initialization
    ###############################################
    centers_index = [generator.randint(0,n)]
    centers = x[centers_index]
    # print('centers: ', centers_index)
    k = 1
    N, d = x.shape
    col = np.array([x for x in range(N)]).reshape(-1,1)
    while k < n_cluster:
        r = generator.rand()
        centers_vector = np.expand_dims(centers, axis=1)
        distances = np.sum((x - centers_vector)**2, axis=2)
        y = np.argmin(distances, axis=0).reshape(-1,1)
        select_distance = distances[y, col].T
        prob = select_distance / np.sum(select_distance, axis=1, keepdims=True)
        cumsum = np.cumsum(prob, axis=1)
        center = np.argmax(cumsum >= r)
        centers_index.append(center)
        centers = x[centers_index]
        k+=1

    return centers_index

# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        
        ###################################################################
        #  Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        
        J = 0
        y = np.zeros(N)
        run = 0
        centers = x[self.centers, :]
        for iteration in range(self.max_iter):
            run = iteration
            centers_vector = np.expand_dims(centers, axis=1)
            distances = np.sum((x - centers_vector)**2, axis=2)
            y = np.argmin(distances, axis=0)
            J_new = np.sum([np.sum((x[y == k] - centers[k]) ** 2) for k in range(self.n_cluster)]) / N
            if np.abs(J - J_new) < self.e:
                break 
            J = J_new
            new_means = np.array([np.average(x[y == k], axis=0) for k in range(self.n_cluster)])
            index = np.where(np.isnan(new_means))
            new_means[index] = centers[index]
            centers = new_means
        return centers, y, run
         
class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        model = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, num_runs = model.fit(x, centroid_func)
        centroid_labels = np.zeros(self.n_cluster)
        for n in range(self.n_cluster):
            index = np.where(membership == n)
            counts = np.bincount(y[index])
            centroid_labels[n] = np.argmax(counts)

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        distances = np.sum(((x - np.expand_dims(self.centroids, axis=1)) ** 2), axis=2)
        membership = np.argmin(distances, axis=0)
        y = self.centroid_labels[membership]
        
        return np.array(y)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    depth = image.shape[0]
    image_reshape = np.array(image.reshape(-1,3))
    for index, feature in enumerate(image_reshape):
        distance = np.linalg.norm(code_vectors - feature, axis=1, keepdims=True) 
        ui_min = np.argmin(distance)
        image_reshape[index] = code_vectors[ui_min].reshape(1,-1)
    return image_reshape.reshape(depth,-1,3)

