import numpy as np

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        # perform "max_iterations" steps of
        # gradient descent with step size "step_size"
        # to minimize perceptron loss

        yn = np.where(y == 0, -1, 1)
        for iter in range(max_iterations):       
            pred = np.dot(w, X.T) + b
            indicator = np.where(pred * yn <= 0, 1, 0)
            w = w + step_size * np.dot(indicator * yn, X) / N
            b = b + step_size * np.sum(indicator * yn) / N

    elif loss == "logistic":
        # perform "max_iterations" steps of
        # gradient descent with step size "step_size"
        # to minimize logistic loss

        for iter in range(max_iterations):
            z = np.dot(X,w) + b
            pred = sigmoid(z)
            error = pred - y
            w = w - step_size / N * np.dot(X.T, error)
            b = b - step_size / N * np.sum(error)

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """
    
    value = 1 / (1 + np.exp(-z))
    return value

def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
    preds = np.sign(np.matmul(X,w) + b)
    preds[preds == -1] = 0
    assert preds.shape == (N,) 
    return preds

def multiclass_train(X, y, C,
                      w0=None, 
                      b0=None,
                      gd_type="sgd",
                      step_size=0.5, 
                      max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
 	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)

    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            # perform "max_iterations" steps of
            # stochastic gradient descent with step size
            # "step_size" to minimize logistic loss.

            z = np.dot(w, X[n].T) + b.T
            z_mean = z - z.max()
            exp_z = np.exp(z_mean)
            softmax = exp_z / np.sum(exp_z)
            softmax[y[n]] -= 1
            w = w - step_size * np.dot(softmax.reshape(-1,1), X[n].reshape(1,-1)) 
            b = b - step_size * softmax 
      
    elif gd_type == "gd":
        # perform "max_iterations" steps of
        # gradient descent with step size "step_size"
        # to minimize logistic loss.

        onehot_yn = np.zeros((N, C))
        for i, value in enumerate(y):
            onehot_yn[i, value] = 1
        for it in range(max_iterations):
            z = np.dot(w,X.T) + b.reshape(-1,1)
            z_mean = z - np.max(z)
            exp_z = np.exp(z_mean)
            exp_z_sum = np.sum(exp_z, axis=0)
            softmax = exp_z / exp_z_sum
            softmax -= onehot_yn.T
            w = w - step_size / N * np.dot(softmax, X)
            b = b - step_size / N * np.sum(softmax, axis=1)
            
    else:
        raise "Undefined algorithm."

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b

def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    b = b.reshape(1, -1)
    probability = np.dot(X,np.transpose(w)) + b
    preds = np.argmax(probability, axis=1)
    assert preds.shape == (N,)
    return preds
