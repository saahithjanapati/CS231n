from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        classes_outside_desired_margin = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
              continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                # update gradients
                loss += margin
                classes_outside_desired_margin += 1

                dW[:,j] += X[i]
        dW[:,y[i]] += -1*classes_outside_desired_margin*X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dW /= num_train # scale by number of items in training set
    dW += 2*W*reg # take care of regularization

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    delta = 1 # hyperparameter for SVM-multiclass loss

    scores = X.dot(W) # matrix of size (N, C)
    # one row for each item in training set
    # each column represents scores for a certain class

    # vector containing the correct class scores for each row in scores
    correct_class_scores = scores[np.arange(X.shape[0]), y].reshape((scores.shape[0], 1))

    # for all wrong classes in each row, set loss to:
    #     - max(0, scores[j] - correct_class_score + delta)
    loss_inter = scores-correct_class_scores + delta
    loss_mat = np.maximum(np.zeros(scores.shape), loss_inter)
    
    # for right class, set to 0 (equivalent to subtracting delta N times, where N is the number of training samples)
    loss_mat[np.arange(X.shape[0]), y] = 0

    # loss = sum of entire matrix
    loss = np.sum(loss_mat)

    # scale by number of training examples
    loss /= X.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # create boolean matrix (1 for classes greater than margin, 0 for classes not greater than margin)
    intermediate_matrix_for_grad = (loss_inter > 0).astype(int)
    
    # sum over columns of intermediate_matrix_for_grad to count how many classes went over the margin
    # subtract one because we don't want to count the actual correct class
    counts = np.sum(intermediate_matrix_for_grad, axis=1) - 1
    intermediate_matrix_for_grad[np.arange(intermediate_matrix_for_grad.shape[0]), y] = -counts

    dW = (X.T).dot(intermediate_matrix_for_grad)
    dW /= X.shape[0] # normalize by number of training elements
    dW += 2*W*reg # take care of regularization
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
