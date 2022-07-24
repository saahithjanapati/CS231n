from builtins import range
import numpy as np
from random import shuffle
from numpy.lib.npyio import savez_compressed
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    for i in range(num_train):
      scores = X[i].dot(W)  # compute predicted scores for training sample
      scores -= np.max(scores) #subtract max of array from all elements to prevent numeric instability issues
      scores = np.exp(scores) # exponetiate all terms in the array
      total = np.sum(scores)
      scores /= total  # normalize probabilities
      correct_class_probability = scores[y[i]]
      loss += -np.log(correct_class_probability)
      
      for class_index in range(scores.shape[0]):
        if class_index == y[i]: # CORRECT CLASS
          dW[:, class_index] += -X[i]+X[i]*(scores[class_index])
        else: # ALL OTHER CLASSES
          dW[:, class_index] += X[i] * scores[class_index]

    # divide by number of training examples
    dW /= num_train 
    loss /= num_train

    # add regularization term
    loss += reg * np.sum(W*W)
    dW += 2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W) # compute predicted scores

    max_vector = np.amax(scores, axis=1).reshape(num_train, 1)
    scores -= max_vector # find max score for each training example
    scores = np.exp(scores) # exponentiate
    sum_vector = np.sum(scores, axis=1).reshape(num_train, 1)

    scores /= sum_vector # divide rows by their sum (softmax)
    
    correct_class_probabilities = scores[np.arange(num_train), y]    
    loss += np.sum(-np.log(correct_class_probabilities))


    # each column in the dW matrix is linear combination of X[i] vectors
    # for correct class, weight = (scores[class_index] - 1)
    # for incorrect classes, weight = (scores[class_index])
    weights = np.zeros((num_train, num_classes))
    weights[np.arange(num_train), y] = -1
    weights = scores + weights
    dW = X.T.dot(weights)

    # divide by number of training examples
    dW /= num_train 
    loss /= num_train

    # add regularization term
    loss += reg * np.sum(W*W)
    dW += 2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
