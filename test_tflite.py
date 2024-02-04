import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config

# Set precision for TensorFlow operations to avoid NumPy warning
np_config.enable_numpy_behavior()

# Load the TFLite model
model_path = 'model/detection.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load an input image for inference
image_path = 'samples/8.png'
image = Image.open(image_path)
# print(image.size)
image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
# print(image.size)
input_image = np.expand_dims(np.array(image) / 255.0, axis=0).astype(input_details[0]['dtype'])
# print(input_image.shape)
input_image = input_image[:, :, :, 0:1]
# print(input_image.shape)

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# Get the output tensors
output_boxes = interpreter.get_tensor(output_details[0]['index'])
output_classes = interpreter.get_tensor(output_details[1]['index'])

print("Detected Class: ", output_classes)
print("Detected Box ", output_boxes)

# top_class = np.argmax(output_classes[0])
# top_box = output_boxes[0][top_class-1]

# print("Detected Class:", top_class)
# print("Bounding Box Coordinates:", top_box)

# Non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=10, iou_threshold=iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    return selected_boxes.numpy(), selected_scores.numpy()

# Decode bounding box coordinates
def decode_boxes(anchors, predictions):
    decoded_boxes = np.zeros_like(predictions)
    decoded_boxes[:, 0] = predictions[:, 0] * anchors[:, 2] + anchors[:, 0]  # ymin
    decoded_boxes[:, 1] = predictions[:, 1] * anchors[:, 3] + anchors[:, 1]  # xmin
    decoded_boxes[:, 2] = predictions[:, 2] * anchors[:, 2] + anchors[:, 0]  # ymax
    decoded_boxes[:, 3] = predictions[:, 3] * anchors[:, 3] + anchors[:, 1]  # xmax
    return decoded_boxes

# Post-process the output to get bounding boxes and classes
anchors = np.array([[0, 0, 1, 1]])  # Replace with your anchor box values

decoded_boxes = decode_boxes(anchors, output_boxes[0])
selected_boxes, selected_scores = non_max_suppression(decoded_boxes, output_classes[0])

# Plot bounding box on the image
def plot_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Convert boxes from normalized coordinates to image coordinates
height, width, _ = np.array(image).shape
selected_boxes[:, [0, 2]] *= height
selected_boxes[:, [1, 3]] *= width

# Plot bounding boxes on the image
plot_boxes(np.array(image), selected_boxes)