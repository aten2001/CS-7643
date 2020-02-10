import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = x.reshape([x.shape[0],-1]).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  x2 = x.reshape(N, D)

  dx2 = np.dot(dout, w.T)
  # N x D
  dw = np.dot(x2.T, dout)
  # D x M
  db = np.dot(dout.T, np.ones(N))
  # M x 1

  dx = np.reshape(dx2, x.shape)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout * (x > 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  (N,C,H,W) = x.shape
  (F,_,HH,WW) = w.shape
  H_p = int(1 + (H + 2 * pad - HH) / stride)
  W_p = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, H_p, W_p))

  for n in range(N):
    x_pad = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
    for f in range(F):
      for h_p in range(H_p):
        for w_p in range(W_p):
          h_1 = h_p * stride
          w_1 = w_p * stride
          window = x_pad[:,h_1:(h_1 + HH), w_1:(w_1 + WW)]
          out[n, f, h_p, w_p] = np.sum(window * w[f,:,:,:]) + b[f]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x,w,b,conv_param = cache
  (N,C,H,W) = x.shape
  (F,_,HH,WW) = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
  dw = np.zeros_like(w)
  db = np.sum(dout, axis=(0,2,3))
  dx_pad = np.zeros_like(x_pad)
  #print(dw.shape)
  Hp = int(1 + (H + 2 * pad - HH) / stride)
  Wp = int(1 + (W + 2 * pad - WW) / stride)
  #print(Wp)
  for n in range(N):
    for f in range(F):
      for c in range(C):
        for hp in range(Hp):
          for wp in range(Wp):
            dx_pad[n, c, hp * stride:(hp * stride + HH), wp * stride:(wp*stride + WW)] += dout[n, f, hp, wp] * w[f, c]
            dw[f, c] += x_pad[n, c, hp * stride:(hp * stride + HH), wp * stride:(wp*stride + WW)] * dout[n, f, hp, wp]
  dx = dx_pad[:, :, pad:-pad, pad:-pad]                  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  (N,C,H,W) = x.shape
  height = pool_param['pool_height']
  width = pool_param['pool_width']
  stride = pool_param['stride']
  H_p = int(1 + (H - height) / stride)
  W_p = int(1 + (H - width) / stride)

  out = np.zeros((N, C, H_p, W_p))
  for n in range(N):
    for h in range(H_p):
      for w in range(W_p):
        h1 = h * stride
        w1 = w * stride
        window = x[n, :, h1:(h1 + height), w1:(w1 + width)]
        out[n,:,h,w] = np.max(window.reshape((C, height * width)), axis=1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  (N,C,H,W) = x.shape
  height = pool_param['pool_height']
  width = pool_param['pool_width']
  stride = pool_param['stride']
  H_m = int(1 + (H - height) / stride)
  W_m = int(1 + (W - width) / stride)

  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
        for hm in range(H_m):
            for wm in range(W_m):
              h1 = hm * stride
              h2 = h1 + height
              w1 = wm * stride
              w2 = w1 + width
              max_index = np.argmax(x[n, c, h1:h2, w1:w2])
              co_x, co_y = np.unravel_index(max_index, (height, width))
              dx[n, c, h1:h2, w1:w2][co_x, co_y] = dout[n, c, hm, wm]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

