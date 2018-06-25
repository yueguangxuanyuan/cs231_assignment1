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
  #这里基于公式直接推导最后的结果 就不用循环慢慢套了
  train_n = X.shape[0];
  pred_y = X.dot(W);
  exp_y = np.exp(pred_y);
  row_sum_exp_y = np.reshape(np.sum(exp_y,axis=1),[train_n,1]);
  softmax_y_matrix = exp_y/row_sum_exp_y;
  softmax_list = softmax_y_matrix[range(train_n),y];
  loss_list = -np.log(softmax_list);
  loss = np.sum(loss_list)/train_n + 0.5*reg*np.sum(W*W);
  
  dw_matrix = softmax_y_matrix;
  dw_matrix[range(train_n),y] -=1;
  dW = X.T.dot(dw_matrix)/train_n + reg*W;
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
  train_n = X.shape[0];
  pred_y = X.dot(W);
  y_max = np.max(pred_y,axis=1);
  pred_y -= np.reshape(y_max,[train_n,1]);
  exp_y = np.exp(pred_y);
  row_sum_exp_y = np.reshape(np.sum(exp_y,axis=1),[train_n,1]);
  
  softmax_y_matrix = exp_y/row_sum_exp_y;
  softmax_list = softmax_y_matrix[range(train_n),y];
  loss_list = -np.log(softmax_list);
  #loss_list=softmax_list;
  loss = np.sum(loss_list)/train_n + 0.5*reg*np.sum(W*W);
  
  dw_matrix = softmax_y_matrix;
  dw_matrix[range(train_n),y] -=1;
  dW = X.T.dot(dw_matrix)/train_n + reg*W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

