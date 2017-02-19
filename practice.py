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
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  out_h = 1 + (H + 2 * pad - HH) / stride
  out_w = 1 + (W + 2 * pad - WW) / stride
  #print "out h, w", out_h, out_w
  out = np.zeros([N, F, out_h, out_w])
  for i in range(N):
      x_i = x[i,:,:,:]
      #print "x_i shape is", x_i.shape
      npad = ((0,0),(pad,pad),(pad,pad))
      #print "npad is", npad
      x_pad = np.pad(x_i, pad_width=npad, mode='constant', constant_values=0)
      #print "x_pad shape is", x_pad.shape
      h_count, w_count = 0,0
      for j in range(0, x_pad.shape[1] - HH + 1, stride):
          w_count = 0
          for k in range(0, x_pad.shape[2] - WW + 1, stride):
              for filt in range(F):
                  x_sub = x_pad[:, j:j+HH, k:k+WW]
                  #print "w_count, h_count", w_count, h_count
                  out[i, filt, h_count, w_count] += np.sum(x_sub * w[filt,:,:,:]) + b[filt]
              w_count += 1
          h_count += 1


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
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  out_h = 1 + (H + 2 * pad - HH) / stride
  out_w = 1 + (W + 2 * pad - WW) / stride
  #print "out h, w", out_h, out_w
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  dx_pad = np.pad(dx, pad_width=((0,0), (0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)
  for i in range(N):
      x_i = x[i,:,:,:]
      #print "x_i shape is", x_i.shape
      npad = ((0,0),(pad,pad),(pad,pad))
      #print "npad is", npad
      x_pad = np.pad(x_i, pad_width=npad, mode='constant', constant_values=0)
      #dx_pad = np.pad(dx_i, pad_width=npad, mode='constant', constant_values=0)
      #print "x_pad shape is", x_pad.shape
      h_count, w_count = 0,0
      for j in range(0, x_pad.shape[1] - HH + 1, stride):
          w_count = 0
          for k in range(0, x_pad.shape[2] - WW + 1, stride):
              for filt in range(F):
                  x_sub = x_pad[:, j:j+HH, k:k+WW]
                  #print "w_count, h_count", w_count, h_count
                  # Compute gradient of out[i, filt, h_count, w_count] = np.sum(window*w[j]) + b[j]
                  dw[filt] += x_sub * dout[i, filt, h_count, w_count]
                  db[filt] += dout[i, filt, h_count, w_count]
                  dx_pad[i, :, j:j+HH, k:k+WW] += w[filt] *  dout[i, filt, h_count, w_count]
              w_count += 1
          h_count += 1
  dx = dx_pad[:,:,pad:-pad,pad:-pad]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db