import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Linear Regression (Ridge Regression) (20 points)
    In this problem, we will solve a linear regression problem using ridge regression
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Linear Regression with Least Square Loss) Let's start with a simpler method (called 'Least Square'). Given a set of training samples (X) and their labels (y), train a linear model (with parameters w) on training samples using least-square regression. 
    ---- Inputs: --------
        * X: the feature matrix of the training samples, a numpy matrix of shape n by p, where X[i] represents the p-dimensional feature vector for the i-th training sample in the training dataset.
        * y: the labels, a numpy vector of length n, y[i] represents the label of the i-th training sample in the training dataset.
    ---- Outputs: --------
        * w: the weights of the linear regression model, a numpy float vector of length p.
    ---- Hints: --------
        * You could use np.linalg.inv() to compute the inverse of a matrix. 
        * You could use @ operator in numpy for matrix multiplication: A@B represents the matrix multiplication between matrices A and B. 
        * You could use A.T to compute the transpose of matrix A. 
        * This problem can be solved using 1 line(s) of code.
'''
#----------------------
def least_square(X, y):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    w = (np.linalg.inv(X.T@X))@(np.dot(X.T, y))
    #########################################
    return w
    #------ (10 points / 20 total points) -----------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        --- for Mac OS ---- 
        python3 -m pytest -v test2.py::test_least_square_10pt
        --- for Windows OS ---- 
        python -m pytest -v test2.py::test_least_square_10pt
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Ridge Regression) Now let's build the ridge regression by adding a regularization term to the least square loss.   Given a set of training samples (X) and their labels (y), train a linear model (with parameters w) on training samples using ridge regression (i.e., least square regression with L2 regularization). 
    ---- Inputs: --------
        * X: the feature matrix of the training samples, a numpy matrix of shape n by p, where X[i] represents the p-dimensional feature vector for the i-th training sample in the training dataset.
        * y: the labels, a numpy vector of length n, y[i] represents the label of the i-th training sample in the training dataset.
        * a: (alpha) the weight of the regularization term in ridge regression, a float scalar.
    ---- Outputs: --------
        * w: the weights of the linear regression model, a numpy float vector of length p.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#----------------------
def ridge_regression(X, y, a=0.0001):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    (H, W) = (X.T @ X).shape
    w = (np.linalg.inv(X.T @ X + a*np.identity(H))) @ (np.dot(X.T, y))
    #########################################
    return w
    #------ (10 points / 20 total points) -----------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        --- for Mac OS ---- 
        python3 -m pytest -v test2.py::test_ridge_regression_10pt
        --- for Windows OS ---- 
        python -m pytest -v test2.py::test_ridge_regression_10pt
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 2: (20 points)
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        --- for Mac OS ---- 
        python3 -m pytest -v test2.py
        --- for Windows OS ---- 
        python -m pytest -v test2.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- test session starts --------------------- 
        * (10 points) test2.py::least_square_10pt  PASSED 
        * (10 points) test2.py::ridge_regression_10pt  PASSED 
        
        ============ 2 passed in 0.004s ======================
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* p:  the number of features of each data sample. 
* n:  the number of training samples in training data set. 
* X:  the feature matrix of the training samples, a numpy matrix of shape n by p, where X[i] represents the p-dimensional feature vector for the i-th training sample in the training dataset. 
* y:  the labels, a numpy vector of length n, y[i] represents the label of the i-th training sample in the training dataset. 
* w:  the weights of the linear regression model, a numpy float vector of length p. 
* a:  (alpha) the weight of the regularization term in ridge regression, a float scalar. 

'''
#--------------------------------------------