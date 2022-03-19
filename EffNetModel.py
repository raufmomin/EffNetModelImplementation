from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, DepthwiseConv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.initializers import HeUniform

def effnet_block(X, numFiltersIn, numFiltersOut, blockNum):

    # Sub-block A
    X = Conv2D(filters = numFiltersIn,
               kernel_size = (1, 1),
               strides = (1, 1),
               padding = 'same',
               activation = None,
               use_bias = False,
               kernel_initializer = HeUniform(seed = 0),
               name = str(blockNum) + 'A-Conv2D')(X)
    X = LeakyReLU(name = str(blockNum) + 'A-LeakyReLU')(X)
    X = BatchNormalization(name = str(blockNum) + 'A-BatchNorm')(X)

    # Sub-block B
    X = DepthwiseConv2D(kernel_size = (1, 3),
                        strides = (1, 1),
                        padding = 'same',
                        activation = None,
                        use_bias = False,
                        kernel_initializer = HeUniform(seed = 0),
                        name = str(blockNum) + 'B-DepthwiseConv2D')(X)
    X = LeakyReLU(name = str(blockNum) + 'B-LeakyReLU')(X)
    X = BatchNormalization(name = str(blockNum) + 'B-BatchNorm')(X)

    # Sub-block C
    X = MaxPool2D(pool_size = (2, 1), strides = (2, 1), padding = 'valid', name = str(blockNum) + 'C-MaxPool2D')(X)  # Separable pooling
    X = DepthwiseConv2D(kernel_size = (3, 1),
                        strides = (1, 1),
                        padding = 'same',
                        activation = None,
                        use_bias = False,
                        kernel_initializer = HeUniform(seed = 0),
                        name = str(blockNum) + 'C-DepthwiseConv2D')(X)
    X = LeakyReLU(name = str(blockNum) + 'C-LeakyReLU')(X)
    X = BatchNormalization(name = str(blockNum) + 'C-BatchNorm')(X)

    # Sub-block D
    X = Conv2D(filters = numFiltersOut,
               kernel_size = (2, 1),
               strides = (1, 2),
               padding = 'same',
               activation = None,
               use_bias = False,
               kernel_initializer = HeUniform(seed = 0),
               name = str(blockNum) + 'D-Conv2D')(X)
    X = LeakyReLU(name = str(blockNum) + 'D-LeakyReLU')(X)
    X = BatchNormalization(name = str(blockNum) + 'D-BatchNorm')(X)

    return X

def EffNet(input_shape, output_classes, include_top = True, weights = None):

    X_input = Input(shape = input_shape, name = '0-Input')

    X_output = effnet_block(X = X_input, numFiltersIn = 32, numFiltersOut = 64, blockNum = 1)
    X_output = effnet_block(X = X_output, numFiltersIn = 64, numFiltersOut = 128, blockNum = 2)
    X_output = effnet_block(X = X_output, numFiltersIn = 128, numFiltersOut = 256, blockNum = 3)

    # Block Top
    if include_top == True:
      X_output = Flatten(name = 'top-flatten')(X_output)
      X_output = Dense(output_classes, activation = 'softmax', name = 'top-dense')(X_output)

    model = Model(inputs = X_input, outputs = X_output, name = 'EffNet')

    # Loading weights
    if weights is not None:
      model.load_weights(weights, by_name = True)

    return model
