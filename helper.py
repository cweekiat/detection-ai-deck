import argparse
import tensorflow as tf

INPUT_SIZE = (324, 244, 1)
# Define the number of classes
NUM_CLASSES = 2   # Balls and Cones
NUM_ANCHORS = 4
FIRST_LAYER_STRIDE = 2

# Function to parse a single example from the TFRecord

def decode_tfrecord_fn(tfrecord):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.VarLenFeature(tf.int64),
        'image/width': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }

    example = tf.io.parse_single_example(tfrecord, feature_description)

    image = tf.image.decode_jpeg(example['image/encoded'], channels=1)
    image = tf.image.resize(image, (324, 244))  # Adjust the size as needed
    image = tf.cast(image, tf.float32) / 255.0

    # Extract bounding box coordinates
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    # Extract labels
    labels = tf.sparse.to_dense(example['image/object/class/label'])

    return image, {'classification_head': labels, 'regression_head': boxes}

# Function to create the TFRecordDataset
def load_dataset(tfrecord_path, batch_size=32,shuffle_buffer_size = 1000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(decode_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def parse_args():
    args = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    args.add_argument("--epochs", dest="epochs", type=int, default=20)
    args.add_argument(
        "--finetune_epochs", dest="finetune_epochs", type=int, default=20
    )
    args.add_argument(
        "--dataset_path",
        metavar="dataset_path",
        help="path to dataset",
        default="training_data",
    )
    args.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    args.add_argument(
        "--image_width", dest="image_width", type=int, default=324
    )
    args.add_argument(
        "--image_height", dest="image_height", type=int, default=244
    )
    args.add_argument(
        "--image_channels", dest="image_channels", type=int, default=1
    )

    return args.parse_args()

'''
def calculate_scale_of_default_boxes(k, m, s_max = 0.9, s_min = 0.2):
    """
    m = number_of_feature_maps
    s_k = s_min + (s_max - s_min) * (k - 1)/(m - 1)
    width_k = s_k * sqrt(aspect_ratio)
    height_k = s_k / sqrt(aspect_ratio)
    """
    return s_min + (s_max - s_min) * (k - 1) / (m - 1)

def generate_default_boxes(feature_map_shapes, number_of_feature_maps, aspect_ratios):
    """
    feature map shapes for VGG: [38, 19, 10, 5, 3, 1]
    """

    assert len(feature_map_shapes) == number_of_feature_maps, 'number of feature maps needs to be {0}'.format(len(feature_map_shapes))
    assert len(feature_map_shapes) == len(aspect_ratios), 'Need aspect ratios for all feature maps'

    prior_boxes = []

    for k, f_k in enumerate(feature_map_shapes):
        s_k = calculate_scale_of_default_boxes(k, m = number_of_feature_maps)
        s_k_prime = np.sqrt(s_k * calculate_scale_of_default_boxes(k + 1, m = 6))
        for i in range(f_k):
            for j in range(f_k):
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                prior_boxes.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    # height, width for numpy
                    prior_boxes.append([cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar)])

    prior_boxes = tf.convert_to_tensor(prior_boxes, dtype=tf.float32)
    return tf.clip_by_value(prior_boxes, clip_value_min = 0., clip_value_max = 1.)


# Adapted from https://gist.github.com/escuccim/d0be49ccfc6084cdc784a67339f130dd
def box_overlap_iou(boxes, gt_boxes):
    """
    Args:
        boxes: shape (total boxes, x_min, y_min, x_max, y_max)
        gt_boxes: shape (1, total label, x_min  y_min, x_max, y_max)

    Returns:
        Tensor with shape (batch_size, total boxes, total label)
    """
    box_x_min, box_y_min, box_x_max, box_y_max = tf.split(boxes, 4, axis = 1)
    gt_boxes_x_min, gt_boxes_y_min, gt_boxes_x_max, gt_boxes_y_max = tf.split(gt_boxes, 4, axis = 2)

    # From https://www.tensorflow.org/api_docs/python/tf/transpose
    intersection_x_min = tf.maximum(box_x_min, tf.transpose(gt_boxes_x_min, perm=[0, 2, 1]))
    intersection_y_min = tf.maximum(box_y_min, tf.transpose(gt_boxes_y_min, perm=[0, 2, 1]))

    intersection_x_max = tf.minimum(box_x_max, tf.transpose(gt_boxes_x_max, perm=[0, 2, 1]))
    intersection_y_max = tf.minimum(box_y_max, tf.transpose(gt_boxes_y_max, perm=[0, 2, 1]))

    # need to take care of boxes that don't overlap at all
    intersection_area = tf.maximum(intersection_x_max - intersection_x_min, 0) * tf.maximum(intersection_y_max - intersection_y_min, 0)

    boxes_areas = (box_x_max - box_x_min) * (box_y_max - box_y_min)
    gt_box_areas = (gt_boxes_x_max - gt_boxes_x_min) * (gt_boxes_y_max - gt_boxes_y_min)

    union = (boxes_areas + tf.transpose(gt_box_areas, perm=[0, 2, 1])) - intersection_area

    return tf.maximum(intersection_area / union, 0)


def match_priors_with_gt(prior_boxes, boxes, gt_boxes, gt_labels, number_of_labels, threshold = 0.5):
    
    """
    prior boxes: (1, number of default boxes, c_x, c_y, w, h)
    boxes: shape (total boxes, x_min, y_min, x_max, y_max)
    gt_boxes: (1, number of labels, x_min, y_min, x_max, y_max)
    gt_labels: (1, 1 label per each gt box)

    0 is background, so the gt_labels is the number of labels in the dataset + 1
    class 0 is reserved.
    """

    # number of rows for the IOU map the is the number of gt_boxes
    IOU_map = box_overlap_iou(boxes, gt_boxes)

    # convert ground boxes labels to box label format
    gt_box_label = convert_to_centre_dimensions_form(gt_boxes)

    # select the box with the highest IOU
    highest_overlap_idx = tf.math.argmax(IOU_map, axis = 1)
    highest_overlap_idx = tf.cast(highest_overlap_idx, tf.int32)
    idx = tf.range(IOU_map.shape[1])
    highest_overlap_idx_map = tf.expand_dims(tf.equal(idx, tf.transpose(highest_overlap_idx)), axis = 0)
    IOU_map = tf.where(tf.transpose(highest_overlap_idx_map, perm=[0,2,1]), tf.constant(1.0), IOU_map)

    # find the column idx with the highest IOU at each row
    max_IOU_idx_per_row = tf.math.argmax(IOU_map, axis = 2)
    # find the max value per row
    max_IOU_per_row = tf.reduce_max(IOU_map, axis = 2)

    # threshold IOU
    max_IOU_above_threshold = tf.greater(max_IOU_per_row, threshold)
    
    # map the gt boxes to the prior boxes with the highest overlap
    gt_box_label_map = tf.gather(gt_box_label, max_IOU_idx_per_row, batch_dims = 1)
    # get the offset, offcet (delta_cx, delta_cy, delta_width, delta_height)
    gt_box_label_map_offsets = calculate_offset_from_gt(gt_box_label_map, prior_boxes)
    # remove from gt_boxes_map where overlap with prior boxes is less than 0.5
    gt_boxes_map_offset_suppressed = tf.where( tf.expand_dims(max_IOU_above_threshold, -1),  
                                        gt_box_label_map_offsets, tf.zeros_like(gt_box_label_map))
    # add a positive condition column for the localization loss
    max_IOU_above_threshold_expand = tf.expand_dims(max_IOU_above_threshold, -1)
    max_IOU_above_threshold_expand = tf.cast(max_IOU_above_threshold_expand, tf.float32)
    gt_boxes_map_offset_suppressed_with_pos_cond = tf.concat([  gt_boxes_map_offset_suppressed, 
                                                                max_IOU_above_threshold_expand ], axis = 2)


    gt_labels_map = tf.gather(gt_labels, max_IOU_idx_per_row, batch_dims = 1)
    # suppress the label where IOU with the gt boxes is < 0.5
    gt_labels_map_suppressed = tf.where( max_IOU_above_threshold, 
                                        gt_labels_map, tf.zeros_like(gt_labels_map))
    gt_labels_one_hot_encoded = tf.one_hot(gt_labels_map_suppressed, number_of_labels)

    return gt_boxes_map_offset_suppressed_with_pos_cond, gt_labels_one_hot_encoded

def calculate_offset_from_gt(gt_boxes_mapped_to_prior, prior_boxes):
    prior_boxes = tf.expand_dims(prior_boxes, axis=0)
    g_j_cx = 10 * (gt_boxes_mapped_to_prior[:, :, 0] - prior_boxes[:, :, 0]) / prior_boxes[:, :, 2]
    g_j_cy = 10 * (gt_boxes_mapped_to_prior[:, :, 1] - prior_boxes[:, :, 1]) / prior_boxes[:, :, 3]
    g_j_w = 5 * tf.math.log(gt_boxes_mapped_to_prior[:, :, 2] / prior_boxes[:, :, 2])
    g_j_h = 5 * tf.math.log(gt_boxes_mapped_to_prior[:, :, 3] / prior_boxes[:, :, 3])

    offset = tf.concat( [ g_j_cx, g_j_cy, g_j_w, g_j_h ] , axis = 0)

    return tf.transpose(tf.expand_dims(offset, axis = 0), perm=[0,2,1])

def convert_to_box_form(boxes):
    """
    Input:
        (number_of_labels, c_x, c_y, width, height)
    Output:
        (number_of_labels, x_min, y_min, x_max, y_max)
    """

    box_coordinates = tf.concat([   boxes[:, :2] - boxes[:, 2:] / 2, 
                                    boxes[:, :2] + boxes[:, 2:] / 2 ], 
                                    axis = 1)

    return tf.clip_by_value(box_coordinates, clip_value_min = 0., clip_value_max = 1.)

def convert_to_centre_dimensions_form(boxes):
    """
    Input:
        boxes: (1, number_of_labels, x_min, y_min, x_max, y_max)
    Output:
        (1, number_of_labels, c_x, c_y, width, height)
    """

    coordinates = tf.concat([
                [
                        (boxes[:, :, 0] + boxes[:, :, 2]) / 2., 
                        (boxes[:, :, 1] + boxes[:, :, 3]) / 2.,
                        boxes[:, :, 2] - boxes[:, :, 0],
                        boxes[:, :, 3] - boxes[:, :, 1]
                ]], axis = 1)
    # need the output in the same format as input, could be imporived
    coordinates = tf.transpose(coordinates, perm=[1,2,0])
    return tf.clip_by_value(coordinates, clip_value_min = 0., clip_value_max = 1.)
'''