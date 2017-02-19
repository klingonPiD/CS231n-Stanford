import numpy as np
from random import shuffle

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
    num_train = X.shape[0]
    for i in range(num_train):
        scores = np.dot(X[i],W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        #stuff to address numberical instability
        #const = np.log(np.max(exp_scores))
        exp_correct_class_score = exp_scores[y[i]]
        loss += -np.log((exp_correct_class_score)/np.sum(exp_scores))
        #print "first arg shape", (X[i,np.newaxis].T).shape
        #print "second arg shape", (np.log(scores)).shape
        p = exp_scores/np.sum(exp_scores)
        ind = np.zeros(p.shape)
        ind[y[i]] = 1.0
        dW += np.dot(X[i,np.newaxis].T,(p-ind).reshape(1,W.shape[1]))
        #dW[:,y[i]] -= np.dot(X[i].T,np.log(exp_scores[y[i]]))
        #dW[:,y[i]] += X[i,:] + np.log(exp_scores[y[i]])


    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*W




    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

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
    num_train = X.shape[0]
    scores = np.dot(X,W)
    #stuff to address numberical instability
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    exp_correct_class_score = exp_scores[range(num_train),y]
    #print "exp scores shape", exp_scores.shape
    #print "exp correct class scores shape", exp_correct_class_score.shape
    #print "const shape is", const.shape
    lossVec = -np.log(exp_correct_class_score/np.sum(exp_scores, axis=1))

    p = exp_scores/np.sum(exp_scores,axis=1)[:,np.newaxis]
    ind = np.zeros(p.shape)
    ind[range(num_train),y] = 1.0
    dW += np.dot(X.T, p-ind)

    loss = np.sum(lossVec)
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

