import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from helper import *

INPUT_SIZE = (224, 224, 1)
# Define the number of classes
NUM_CLASSES = 2   # Balls and Cones
NUM_ANCHORS = 4
FIRST_LAYER_STRIDE = 2
NUM_DEFAULT_BOXES = 4 # Number of default boxes per feature map location
epochs = 5

def mobilenetv2_ssd(INPUT_SIZE=(224, 224, 1), NUM_CLASSES=2, NUM_ANCHORS=4, NUM_DEFAULT_BOXES=4):
    # Input layer for the grayscale image
    input_layer = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]))

    # Separable Convolution and Resizing
    x = layers.SeparableConv2D(
        filters=3,
        kernel_size=1,
        activation=None,
        strides=FIRST_LAYER_STRIDE,)(input_layer)

    x = layers.experimental.preprocessing.Resizing(96, 96, interpolation="bilinear")(x)

    # MobileNetV2 backbone
    global base_model
    base_model = tf.keras.applications.MobileNetV2(
            input_shape=(96,96,3), 
            include_top=False,
            weights='imagenet',
            alpha=0.35
            )
    base_model.trainable = False

    # Additional feature extraction layers
    x = base_model(x)

    classification_head = Sequential([
        layers.SeparableConv2D(filters=32, kernel_size=3, activation="relu", padding='same'),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(units=NUM_CLASSES, activation="sigmoid")
    ], name='classification_head')

    # regression_head = Sequential([
    #     # layers.Conv2D(NUM_ANCHORS * 4, (3, 3), padding='same', activation='linear'),
    #     Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'),        
    #     Conv2D(128, (3, 3), padding='same', activation='relu'),
    #     layers.Flatten(name='flatten_regression'),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(4, activation='linear')  # 4 for (x, y, width, height)
    # ], name='regression_head')

    regression_head = Sequential([
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='sigmoid')  # 4 for (x, y, width, height)
    ], name='regression_head')

    # Create model with input and output layers
    model = tf.keras.Model(inputs=input_layer, outputs=[classification_head(x), regression_head(x)])

    return model


if __name__ == "__main__":
    args = parse_args()
    ROOT_PATH = (
        f"{os.path.abspath(os.curdir)}"
    )
    print(ROOT_PATH)

    DATASET_PATH = f"{ROOT_PATH}{args.dataset_path}"
    if not os.path.exists(DATASET_PATH):
        ROOT_PATH = "./"
        DATASET_PATH = args.dataset_path
    if not os.path.exists(DATASET_PATH):
        raise ValueError(f"Dataset path '{DATASET_PATH}' does not exist.")

    tfrecord_path_train = 'training_data/train.tfrecord'
    tfrecord_path_val = 'training_data/valid.tfrecord'

    # Load datasets
    train_dataset = load_dataset(tfrecord_path_train, batch_size=32)
    validation_dataset = load_dataset(tfrecord_path_val, batch_size=32)
    print('DataLoaded')

    # Create an input tensor
    # inputs = tf.keras.Input(shape=INPUT_SIZE)

    # Build MobileNetV2 SSD model
    ssd_model = mobilenetv2_ssd(INPUT_SIZE, NUM_CLASSES, NUM_ANCHORS)
    # ssd_model = buildSSD_VGG16(2)
    ssd_model.summary()

    # Define optimizer, loss functions, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    classification_loss = tf.keras.losses.CategoricalCrossentropy()
    regression_loss = tf.keras.losses.MeanSquaredError()
    metrics = ['accuracy']

    # Compile the model
    ssd_model.compile(optimizer=optimizer,
                    loss=[classification_loss, regression_loss],
                    metrics=metrics)

    raw_dataset = tf.data.TFRecordDataset('training_data/train.tfrecord')
    for raw_record in raw_dataset.take(5):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # print(example)


    # Train the model
    history = ssd_model.fit(train_dataset,
                epochs=epochs,
                validation_data=validation_dataset)

    # Fine-tune the model
    print("Number of layers in the base model: ", len(base_model.layers))

    base_model.trainable = True
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    ssd_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=[classification_loss, regression_loss],
        metrics=metrics,
    )

    ssd_model.summary()

    history = ssd_model.fit(train_dataset,
                epochs=epochs,
                validation_data=validation_dataset)


    # Convert to TensorFlow lite
    converter = tf.lite.TFLiteConverter.from_keras_model(ssd_model)
    tflite_model = converter.convert()

    with open(f"{ROOT_PATH}/model/detection.tflite", "wb") as f:
        f.write(tflite_model)

'''
    # Convert to quantized TensorFlow Lite
    def representative_data_gen():
        dataset_list = tf.data.Dataset.list_files(DATASET_PATH + "/*/*/*")
        for i in range(100):
            image = next(iter(dataset_list))
            image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(image, channels=1)
            image = tf.image.resize(
                image, [args.image_width, args.image_height]
            )
            image = tf.cast(image, tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]

    converter = tf.lite.TFLiteConverter.from_keras_model(ssd_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with open(
        f"{ROOT_PATH}/model/classification_q.tflite", "wb"
    ) as f:
        f.write(tflite_model)

    batch_images, batch_labels = next(val_generator)

    logits = ssd_model(batch_images)
    prediction = np.argmax(logits, axis=1)
    truth = np.argmax(batch_labels, axis=1)

    keras_accuracy = tf.keras.metrics.Accuracy()
    keras_accuracy(prediction, truth)

    print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

    def set_input_tensor(interpreter, input):
        input_details = interpreter.get_input_details()[0]
        tensor_index = input_details["index"]
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = input

    def classify_image(interpreter, input):
        set_input_tensor(interpreter, input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details["index"])
        # Outputs from the TFLite model are uint8, so we dequantize the results:
        scale, zero_point = output_details["quantization"]
        output = scale * (output - zero_point)
        top_1 = np.argmax(output)
        return top_1

    interpreter = tf.lite.Interpreter(
        f"{ROOT_PATH}/model/classification_q.tflite"
    )
    interpreter.allocate_tensors()

    # Collect all inference predictions in a list
    batch_prediction = []
    batch_truth = np.argmax(batch_labels, axis=1)

    for i in range(len(batch_images)):
        prediction = classify_image(interpreter, batch_images[i])
        batch_prediction.append(prediction)

    # Compare all predictions to the ground truth
    tflite_accuracy = tf.keras.metrics.Accuracy()
    tflite_accuracy(batch_prediction, batch_truth)
    print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))

'''