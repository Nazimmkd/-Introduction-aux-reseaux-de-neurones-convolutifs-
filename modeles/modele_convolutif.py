import numpy as np

def convolution2d(X, K, stride=1, padding=0):
    """
    Convolution 2D pour une image en niveaux de gris.
    X: (H, W)
    K: (KH, KW)
    Retourne: (H', W')
    """
    H, W = X.shape
    KH, KW = K.shape
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding)), mode='constant')
    else:
        X_padded = X
    H_p, W_p = X_padded.shape
    out_H = (H_p - KH) // stride + 1
    out_W = (W_p - KW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.sum(X_padded[i*stride:i*stride+KH, j*stride:j*stride+KW] * K)
    return out

def convolution_rgb(X, K, stride=1, padding=0):
    """
    Convolution pour image RGB.
    X: (H, W, 3)
    K: (KH, KW, 3)
    Retourne: (H', W')
    """
    H, W, C = X.shape
    KH, KW, _ = K.shape
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        X_padded = X
    H_p, W_p, _ = X_padded.shape
    out_H = (H_p - KH) // stride + 1
    out_W = (W_p - KW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.sum(X_padded[i*stride:i*stride+KH, j*stride:j*stride+KW, :] * K)
    return out

def max_pooling(X, pool_size=2, stride=2):
    """
    Max pooling 2D.
    X: (H, W)
    Retourne: (H', W')
    """
    H, W = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.max(X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
    return out

def flatten(X):
    """
    Aplatit le volume en vecteur.
    X: (H, W, C)
    Retourne: (H*W*C,)
    """
    return X.flatten()

# Pour multiple filtres
def conv_layer(X, filters, biases, stride=1, padding=0):
    """
    Couche de convolution avec multiple filtres.
    X: (H, W, C_in)
    filters: (num_filters, KH, KW, C_in)
    biases: (num_filters,)
    Retourne: (H', W', num_filters)
    """
    num_filters, KH, KW, C_in = filters.shape
    H, W, C = X.shape
    assert C == C_in
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        X_padded = X
    H_p, W_p, _ = X_padded.shape
    out_H = (H_p - KH) // stride + 1
    out_W = (W_p - KW) // stride + 1
    out = np.zeros((out_H, out_W, num_filters))
    for f in range(num_filters):
        for i in range(out_H):
            for j in range(out_W):
                out[i, j, f] = np.sum(X_padded[i*stride:i*stride+KH, j*stride:j*stride+KW, :] * filters[f]) + biases[f]
    return out

def pool_layer(X, pool_size=2, stride=2):
    """
    Couche de max pooling.
    X: (H, W, C)
    Retourne: (H', W', C)
    """
    H, W, C = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W, C))
    for c in range(C):
        out[:, :, c] = max_pooling(X[:, :, c], pool_size, stride)
    return out

# Initialisation des paramètres
def init_conv_filters(num_filters, KH, KW, C_in):
    return np.random.randn(num_filters, KH, KW, C_in) * 0.01

def init_conv_biases(num_filters):
    return np.zeros(num_filters)

# Pour la densification (fully connected)
def dense_forward(X, W, b):
    return np.dot(W, X) + b

# Softmax pour la sortie du CNN
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def init_cnn():
    # Première conv : 64 filtres 3x3x3
    conv1_filters = init_conv_filters(64, 3, 3, 3)
    conv1_biases = init_conv_biases(64)
    
    # Deuxième conv : 64 filtres 3x3x64
    conv2_filters = init_conv_filters(64, 3, 3, 64)
    conv2_biases = init_conv_biases(64)
    
    # Troisième conv : 64 filtres 3x3x64
    conv3_filters = init_conv_filters(64, 3, 3, 64)
    conv3_biases = init_conv_biases(64)
    
    # Couche dense : 4096 -> 10
    W_dense = np.random.randn(10, 4096) * np.sqrt(1 / 4096)
    b_dense = np.zeros((10, 1))
    
    params = {
        'conv1_filters': conv1_filters, 'conv1_biases': conv1_biases,
        'conv2_filters': conv2_filters, 'conv2_biases': conv2_biases,
        'conv3_filters': conv3_filters, 'conv3_biases': conv3_biases,
        'W_dense': W_dense, 'b_dense': b_dense
    }
    return params

def cnn_forward(X_batch, params):
    """
    Passe avant pour le CNN.
    X_batch: (batch_size, 32, 32, 3) - lot d'images
    """
    batch_size = X_batch.shape[0]
    
    # Conv1: 64 filtres 3x3x3, padding=1 pour garder 32x32
    conv1_out = np.zeros((batch_size, 32, 32, 64))
    for b in range(batch_size):
        conv1_out[b] = conv_layer(X_batch[b], params['conv1_filters'], params['conv1_biases'], padding=1)
    
    # Conv2: 64 filtres 3x3x64, padding=1
    conv2_out = np.zeros((batch_size, 32, 32, 64))
    for b in range(batch_size):
        conv2_out[b] = conv_layer(conv1_out[b], params['conv2_filters'], params['conv2_biases'], padding=1)
    
    # Pool1: max pooling 2x2 -> 16x16x64
    pool1_out = np.zeros((batch_size, 16, 16, 64))
    for b in range(batch_size):
        pool1_out[b] = pool_layer(conv2_out[b])
    
    # Conv3: 64 filtres 3x3x64, padding=1 -> 16x16x64
    conv3_out = np.zeros((batch_size, 16, 16, 64))
    for b in range(batch_size):
        conv3_out[b] = conv_layer(pool1_out[b], params['conv3_filters'], params['conv3_biases'], padding=1)
    
    # Pool2: max pooling 2x2 -> 8x8x64
    pool2_out = np.zeros((batch_size, 8, 8, 64))
    for b in range(batch_size):
        pool2_out[b] = pool_layer(conv3_out[b])
    
    # Conv4: wait, the PDF has 3 conv, but let's check.
    # PDF: Conv, Conv, Pool, Conv, Pool, Conv
    # So after pool2, another conv?
    # "Convolution par 64 filtres 3D" after second pool? No.
    # Let's count:
    # 1. Convolution par 64 filtres (color)
    # 2. Convolution par 64 filtres 3D
    # 3. Max-Pooling
    # 4. Convolution par 64 filtres 3D
    # 5. Max-Pooling
    # 6. Convolution par 64 filtres 3D
    # So 3 conv, 2 pool.
    # After second pool, no more conv.
    # So flatten pool2_out -> 8x8x64 = 4096
    
    # Aplatissement
    flat = np.zeros((batch_size, 4096))
    for b in range(batch_size):
        flat[b] = flatten(pool2_out[b])
    
    # Dense
    Z_dense = dense_forward(flat.T, params['W_dense'], params['b_dense'])  # (10, batch_size)
    A_out = softmax(Z_dense)
    
    cache = {
        'conv1_out': conv1_out,
        'conv2_out': conv2_out,
        'pool1_out': pool1_out,
        'conv3_out': conv3_out,
        'pool2_out': pool2_out,
        'flat': flat,
        'Z_dense': Z_dense,
        'A_out': A_out
    }
    return A_out, cache

def cnn_backprop(X_batch, Y_batch, cache, params):
    """
    Rétropropagation pour le CNN.
    """
    batch_size = X_batch.shape[0]
    
    # Gradient de la softmax
    dZ_dense = cache['A_out'] - Y_batch  # (10, batch_size)
    
    # Gradients de la couche dense
    dW_dense = (1/batch_size) * np.dot(dZ_dense, cache['flat'])
    db_dense = (1/batch_size) * np.sum(dZ_dense, axis=1, keepdims=True)
    
    # Gradient vers l'aplatissement
    dflat = np.dot(params['W_dense'].T, dZ_dense).T  # (batch_size, 4096)
    
    # Remettre en forme vers pool2_out
    dpool2_out = dflat.reshape((batch_size, 8, 8, 64))
    
    # Rétroprop conv3
    dconv3_out = np.zeros_like(cache['conv3_out'])
    for b in range(batch_size):
        for c in range(64):
            dconv3_out[b, :, :, c] = max_pool_backprop(cache['conv3_out'][b, :, :, c], dpool2_out[b, :, :, c], pool_size=2, stride=2)
    
    # Rétroprop conv3
    dpool1_out = np.zeros_like(cache['pool1_out'])
    dconv3_filters = np.zeros_like(params['conv3_filters'])
    dconv3_biases = np.zeros_like(params['conv3_biases'])
    for b in range(batch_size):
        dpool1_out_b, dconv3_filters_b, dconv3_biases_b = conv_backprop(cache['pool1_out'][b], cache['conv3_out'][b], dconv3_out[b], params['conv3_filters'], params['conv3_biases'], padding=1)
        dpool1_out[b] = dpool1_out_b
        dconv3_filters += dconv3_filters_b
        dconv3_biases += dconv3_biases_b
    dconv3_filters /= batch_size
    dconv3_biases /= batch_size
    
    # Rétroprop pool1
    dconv2_out = np.zeros_like(cache['conv2_out'])
    for b in range(batch_size):
        for c in range(64):
            dconv2_out[b, :, :, c] = max_pool_backprop(cache['conv2_out'][b, :, :, c], dpool1_out[b, :, :, c], pool_size=2, stride=2)
    
    # Rétroprop conv2
    dconv1_out = np.zeros_like(cache['conv1_out'])
    dconv2_filters = np.zeros_like(params['conv2_filters'])
    dconv2_biases = np.zeros_like(params['conv2_biases'])
    for b in range(batch_size):
        dconv1_out_b, dconv2_filters_b, dconv2_biases_b = conv_backprop(cache['conv1_out'][b], cache['conv2_out'][b], dconv2_out[b], params['conv2_filters'], params['conv2_biases'], padding=1)
        dconv1_out[b] = dconv1_out_b
        dconv2_filters += dconv2_filters_b
        dconv2_biases += dconv2_biases_b
    dconv2_filters /= batch_size
    dconv2_biases /= batch_size
    
    # Rétroprop conv1
    dconv1_filters = np.zeros_like(params['conv1_filters'])
    dconv1_biases = np.zeros_like(params['conv1_biases'])
    for b in range(batch_size):
        _, dconv1_filters_b, dconv1_biases_b = conv_backprop(X_batch[b], cache['conv1_out'][b], dconv1_out[b], params['conv1_filters'], params['conv1_biases'], padding=1)
        dconv1_filters += dconv1_filters_b
        dconv1_biases += dconv1_biases_b
    dconv1_filters /= batch_size
    dconv1_biases /= batch_size
    
    grads = {
        'dconv1_filters': dconv1_filters, 'dconv1_biases': dconv1_biases,
        'dconv2_filters': dconv2_filters, 'dconv2_biases': dconv2_biases,
        'dconv3_filters': dconv3_filters, 'dconv3_biases': dconv3_biases,
        'dW_dense': dW_dense, 'db_dense': db_dense
    }
    return grads

def max_pool_backprop(input, grad_output, pool_size=2, stride=2):
    """
    Rétroprop pour max pool.
    input: (H, W)
    grad_output: (H', W')
    """
    H, W = input.shape
    H_out, W_out = grad_output.shape
    grad_input = np.zeros_like(input)
    for i in range(H_out):
        for j in range(W_out):
            window = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            max_val = np.max(window)
            mask = (window == max_val)
            grad_input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size] += mask * grad_output[i, j]
    return grad_input

def conv_backprop(input, output, grad_output, filters, biases, padding=0):
    """
    Rétroprop pour la couche de conv.
    input: (H_in, W_in, C_in)
    output: (H_out, W_out, num_filters)
    grad_output: (H_out, W_out, num_filters)
    filters: (num_filters, KH, KW, C_in)
    biases: (num_filters,)
    """
    num_filters, KH, KW, C_in = filters.shape
    H_in, W_in, C = input.shape
    H_out, W_out, _ = grad_output.shape
    
    # Pad the input
    input_padded = np.pad(input, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    # Gradient pour les biais
    db = np.sum(grad_output, axis=(0,1))  # (num_filters,)
    
    # Gradient pour les filtres
    df = np.zeros_like(filters)
    for f in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                input_region = input_padded[i:i+KH, j:j+KW, :]
                df[f] += grad_output[i, j, f] * input_region
    
    # Gradient pour l'entrée
    dinput_padded = np.zeros_like(input_padded)
    for f in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                dinput_padded[i:i+KH, j:j+KW, :] += grad_output[i, j, f] * filters[f]
    
    # Remove padding from dinput
    dinput = dinput_padded[padding:-padding, padding:-padding, :]
    
    return dinput, df, db