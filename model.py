import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Conv2D, BatchNormalization, ReLU, Add, DepthwiseConv2D, Dropout
from tensorflow.keras.models import Model

# Custom function to create a MobileNetV2-like block with unique names
def mobilenet_block(inputs, filters, prefix, stride = 1):

    x = DepthwiseConv2D((3, 3), strides = stride, padding = 'same', name = f'{prefix}_Depthwise')(inputs)
    x = BatchNormalization(name = f'{prefix}_Depthwise_BN')(x)
    x = ReLU(name = f'{prefix}_Depthwise_ReLU')(x)

    x = Conv2D(filters, (3, 3), strides=stride, padding='same', name=f'{prefix}_Conv')(inputs)
    x = BatchNormalization(name=f'{prefix}_BN')(x)
    x = ReLU(name=f'{prefix}_ReLU')(x)

    return x

def build_model(input_shape_mfcc, input_shape_spec, num_classes, num_blocks = 6):

    # Input for MFCC
    mfcc_input = Input(shape = input_shape_mfcc, name = "mfcc_input")


    x_mfcc = mfcc_input
    for i in range(num_blocks):
        x_mfcc = mobilenet_block(x_mfcc, 32 * (i + 1), f"mfcc_block{i + 1}")
    mfcc_features = GlobalAveragePooling2D(name = "global_avg_pool_mfcc")(x_mfcc)

    '''
    #Manually create a custom MobileNetV2-like model for MFCC
    x_mfcc = mobilenet_block(mfcc_input, 32, "mfcc_block1")
    x_mfcc = mobilenet_block(x_mfcc, 64, "mfcc_block2")
    mfcc_features = GlobalAveragePooling2D(name="global_avg_pool_mfcc")(x_mfcc)
    '''

    # Input for Spectrogram
    spec_input = Input(shape = input_shape_spec, name = "spec_input")
    x_spec = spec_input
    for i in range(num_blocks):
        x_spec = mobilenet_block(x_spec, 32 * (i + 1), f"spec_block{i + 1}")
    spec_features = GlobalAveragePooling2D(name = "global_avg_pool_spec")(x_spec)

    '''
    # Manually create a custom MobileNetV2-like model for Spectrogram
    x_spec = mobilenet_block(spec_input, 32, "spec_block1")
    x_spec = mobilenet_block(x_spec, 64, "spec_block2")
    spec_features = GlobalAveragePooling2D(name="global_avg_pool_spec")(x_spec)
    '''

    # Combine both features
    combined = Concatenate(name = "combined_features")([mfcc_features, spec_features])
    x = Dense(128, activation = 'relu', name = "dense_combined")(combined)
    x = Dropout(0.4, name = "dropout_combined")(x)
    output = Dense(num_classes, activation = 'softmax', name = "output_layer")(x)

    model = Model(inputs = [mfcc_input, spec_input], outputs = output)

    # Debug function to list all layer names
    print("\n🔎 Model Layers:")
    for layer in model.layers:
        output_shape = getattr(layer, 'output_shape', 'N/A')
        print(f"- {layer.name}: {output_shape}")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
