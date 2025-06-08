import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, SeparableConv2D, LSTM, Dense, Flatten, Dropout, Permute, Reshape, Multiply, Add, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.constraints import MaxNorm

# --- Shared Utility Functions ---
def square(x):
    return tf.square(x)

def log(x):
    return tf.math.log(tf.clip_by_value(x, 1e-7, 1.)) # Clip to avoid log(0)

# --- Architecture 1: Hybrid CNN-LSTM Network ---
def build_hybrid_cnn_lstm(input_shape, num_classes, dropout_rate=0.5, conv_filters=32, lstm_units=64):
    """
    Builds a Hybrid CNN-LSTM model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of the input EEG data (channels, time_points, 1).
                             MNE epochs usually yield (n_epochs, n_channels, n_times).
                             For Keras Conv2D, it should be (batch, height, width, channels).
                             Here, we treat channels as height, time_points as width.
        num_classes (int): Number of output classes (e.g., 6 for game commands).
        dropout_rate (float): Dropout regularization rate.
        conv_filters (int): Number of filters in the convolutional layers.
        lstm_units (int): Number of units in the LSTM layers.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    n_channels, n_time_points = input_shape # Assuming input_shape is (channels, time_points)
    
    # Keras Conv2D expects (batch, height, width, channels). For EEG, we can use:
    # (batch, n_channels, n_time_points, 1) or (batch, 1, n_channels, n_time_points)
    # Let's use (batch, n_channels, n_time_points, 1) and apply conv over time/channels.
    input_layer = Input(shape=(n_channels, n_time_points, 1))

    # Spatial-Temporal Convolutional Block
    conv1 = Conv2D(conv_filters, (1, 50), # Kernel across time, for each channel
                   padding='same',
                   data_format='channels_last',
                   kernel_constraint=MaxNorm(1.),
                   input_shape=(n_channels, n_time_points, 1))(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = AveragePooling2D((1, 4), data_format='channels_last')(conv1) # Pool over time
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = Conv2D(conv_filters * 2, (n_channels, 1), # Kernel across channels, for each time point
                   padding='valid', # valid means no padding, kernel must match channels
                   data_format='channels_last',
                   kernel_constraint=MaxNorm(1.))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = AveragePooling2D((1, 4), data_format='channels_last')(conv2) # Pool over time
    conv2 = Dropout(dropout_rate)(conv2)

    # Reshape for LSTM
    # Output shape of conv2 is (batch, 1, new_time_points, conv_filters*2)
    # We need (batch, time_steps, features) for LSTM
    # Remove the channel dimension (dim 1) and flatten the last two dims into features
    # Reshape: (batch, new_time_points, conv_filters*2)
    squeezed = tf.squeeze(conv2, axis=1) # Remove the 1-dimension (channels_last) if it's (batch, 1, T, F)
    # If conv2 is (batch, H, W, C), and after pooling, H becomes 1, W becomes T', C becomes F'
    # Then squeezed will be (batch, T', F')
    
    # If squeezed doesn't work as expected, manually flatten and reshape
    if len(squeezed.shape) == 4: # Still (batch, 1, T', F')
        features_for_lstm = Reshape((-1, conv_filters * 2))(squeezed) # (batch, new_time_points, conv_filters*2)
    else: # Already (batch, T', F')
        features_for_lstm = squeezed


    # Bidirectional LSTM Layers
    lstm1 = tf.keras.layers.Bidirectional(LSTM(lstm_units, return_sequences=True))(features_for_lstm)
    lstm1 = Dropout(dropout_rate)(lstm1)
    lstm2 = tf.keras.layers.Bidirectional(LSTM(lstm_units))(lstm1) # Last LSTM returns single sequence
    lstm2 = Dropout(dropout_rate)(lstm2)

    # Classification Head
    flatten = Flatten()(lstm2)
    dense1 = Dense(lstm_units // 2, activation='relu', kernel_constraint=MaxNorm(0.5))(flatten)
    dense1 = Dropout(dropout_rate)(dense1)
    output_layer = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

# --- Architecture 2: Attention-Enhanced EEGNet ---
def build_attention_eegnet(input_shape, num_classes, F1=8, D=2, F2=16, dropout_rate=0.5, kern_length=64):
    """
    Builds an Attention-Enhanced EEGNet model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of the input EEG data (channels, time_points).
                             MNE epochs usually yield (n_epochs, n_channels, n_times).
                             For Keras, it should be (batch, n_channels, n_time_points, 1).
        num_classes (int): Number of output classes.
        F1 (int): Number of temporal filters.
        D (int): Depth multiplier for depthwise convolution.
        F2 (int): Number of pointwise filters (F2 = F1 * D).
        dropout_rate (float): Dropout regularization rate.
        kern_length (int): Length of the temporal convolution kernel.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    n_channels, n_time_points = input_shape
    
    # Input layer expects (batch, channels, time_points, 1)
    input_layer = Input(shape=(n_channels, n_time_points, 1))

    # Block 1: Temporal Convolution
    block1 = Conv2D(F1, (1, kern_length),
                    padding='same',
                    data_format='channels_last',
                    use_bias=False)(input_layer)
    block1 = BatchNormalization()(block1)

    # Block 2: Depthwise Convolution
    # This applies a separate spatial filter to each feature map from Block 1
    block2 = DepthwiseConv2D((n_channels, 1),
                             depth_multiplier=D,
                             data_format='channels_last',
                             use_bias=False,
                             depthwise_constraint=MaxNorm(1.))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation(square)(block2) # Custom activation for squaring
    block2 = AveragePooling2D((1, 4), data_format='channels_last')(block2)
    block2 = Activation(log)(block2) # Custom activation for log
    block2 = Dropout(dropout_rate)(block2)

    # Block 3: Separable Convolution (Pointwise and Temporal)
    # Pointwise convolution across feature maps, then temporal convolution
    block3 = SeparableConv2D(F2, (1, 16), # Pointwise then temporal
                             padding='same',
                             data_format='channels_last',
                             use_bias=False)(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3) # ELU is common in EEGNet
    block3 = AveragePooling2D((1, 8), data_format='channels_last')(block3)
    block3 = Dropout(dropout_rate)(block3)

    # --- Squeeze-and-Excitation Attention Block ---
    # Apply attention after the main feature extraction blocks
    squeeze = GlobalAveragePooling2D(data_format='channels_last')(block3) # (batch, F2)
    
    excitation = Dense(F2 // 8, activation='relu')(squeeze) # Reduce dimensionality
    excitation = Dense(F2, activation='sigmoid')(excitation) # Scale back to F2, sigmoid for weights

    # Reshape excitation to (batch, 1, 1, F2) to broadcast for element-wise multiplication
    excitation = Reshape((1, 1, F2))(excitation)
    
    # Apply attention weights to block3
    attention_output = Multiply()([block3, excitation])

    # Classification Head
    flatten = Flatten()(attention_output) # Flatten the attention_output
    output_layer = Dense(num_classes, activation='softmax', kernel_constraint=MaxNorm(0.25))(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model


if __name__ == "__main__":
    # Example usage:
    # Assuming input data shape: (batch_size, n_channels, n_time_points)
    # For MNE epochs: n_channels = 6, n_time_points can be 500 * (epoch_tmax - epoch_tmin)
    # Let's say we have 6 channels and 500 time points (1 second at 500 Hz)
    
    N_CHANNELS = 6
    N_TIME_POINTS = 500 # e.g., 1 second epoch at 500Hz
    NUM_CLASSES = 6 # Push, Pull, Left, Right, Up, Down

    print("--- Building Hybrid CNN-LSTM Model ---")
    cnn_lstm_model = build_hybrid_cnn_lstm(input_shape=(N_CHANNELS, N_TIME_POINTS), num_classes=NUM_CLASSES)
    
    print("\n--- Building Attention-Enhanced EEGNet Model ---")
    eegnet_model = build_attention_eegnet(input_shape=(N_CHANNELS, N_TIME_POINTS), num_classes=NUM_CLASSES)

    # You can now save these models or use them for training.
    # e.g., cnn_lstm_model.save('cnn_lstm_model.h5')
