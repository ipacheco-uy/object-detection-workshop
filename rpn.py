import click
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from PIL import Image, ImageDraw, ImageFont

from resnet import resnet_v1_101, resnet_v1_101_tail


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

ANCHOR_BASE_SIZE = 256
ANCHOR_RATIOS = [0.5, 1, 2]
ANCHOR_SCALES = [0.125, 0.25, 0.5, 1, 2]

OUTPUT_STRIDE = 16

PRE_NMS_TOP_N = 12000
POST_NMS_TOP_N = 2000
NMS_THRESHOLD = 0.7

CLASS_NMS_THRESHOLD = 0.5
TOTAL_MAX_DETECTIONS = 300


def open_image(path):
    path = os.path.expanduser(path)
    raw_image = Image.open(path)
    image = np.expand_dims(raw_image.convert('RGB'), axis=0)
    return image


def to_image(image_array):
    return Image.fromarray(np.squeeze(image_array, axis=0))


def draw_rectangle(draw, coordinates, color, width=1):
    outline = tuple(color + [255])

    for i in range(width):
        coords = [
            coordinates[0] - i,
            coordinates[1] - i,
            coordinates[2] + i,
            coordinates[3] + i,
        ]
        if i == 0:
            draw.rectangle(coords, outline=outline)
        else:
            draw.rectangle(coords, outline=outline)


def draw_bboxes(image_array, objects):
    # Receives a numpy array. Translate into a PIL image.
    # TODO: Make optional, or more robust.
    image = to_image(image_array)

    # Open as 'RGBA' in order to draw translucent boxes.
    draw = ImageDraw.Draw(image, 'RGBA')
    for obj in objects:
        color = [255, 0, 0]
        draw_rectangle(draw, obj, color, width=2)

    return image


def draw_bboxes_with_labels(image_array, classes, objects, labels):
    # Receives a numpy array. Translate into a PIL image.
    # TODO: Make optional, or more robust.
    image = to_image(image_array)

    # Open as 'RGBA' in order to draw translucent boxes.
    draw = ImageDraw.Draw(image, 'RGBA')
    for obj, label in zip(objects, labels):
        color = [255, 0, 0]
        draw_rectangle(draw, obj, color, width=3)

        # Draw the object's label.
        font = ImageFont.truetype(
            # TODO: Make this OS-agnostic.
            '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf', 14
        )

        text = classes[label]
        label_w, label_h = font.getsize(text)
        background_coords = [
            obj[0] + 1,
            obj[1],
            obj[0] + label_w + 2,
            obj[1] + label_h + 3,
        ]
        draw.rectangle(background_coords, fill=tuple(color + [255]))

        draw.text(obj[:2], text, font=font)

    return image


def clip_boxes(bboxes, imshape):
    """
    Clips bounding boxes to image boundaries based on image shape.

    Args:
        bboxes: Tensor with shape (num_bboxes, 4)
            where point order is x1, y1, x2, y2.

        imshape: Tensor with shape (2, )
            where the first value is height and the next is width.

    Returns
        Tensor with same shape as bboxes but making sure that none
        of the bboxes are outside the image.
    """
    with tf.name_scope('BoundingBoxTransform/clip_bboxes'):
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        imshape = tf.cast(imshape, dtype=tf.float32)

        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = imshape[1]
        height = imshape[0]
        x1 = tf.maximum(tf.minimum(x1, width - 1.0), 0.0)
        x2 = tf.maximum(tf.minimum(x2, width - 1.0), 0.0)

        y1 = tf.maximum(tf.minimum(y1, height - 1.0), 0.0)
        y2 = tf.maximum(tf.minimum(y2, height - 1.0), 0.0)

        bboxes = tf.concat([x1, y1, x2, y2], axis=1)

        return bboxes


def get_width_upright(bboxes):
    """
    TODO: Docstring.
    """
    with tf.name_scope('BoundingBoxTransform/get_width_upright'):
        bboxes = tf.cast(bboxes, tf.float32)
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = x2 - x1 + 1.
        height = y2 - y1 + 1.

        # Calculate up right point of bbox (urx = up right x)
        urx = x1 + .5 * width
        ury = y1 + .5 * height

        return width, height, urx, ury


def change_order(bboxes):
    """Change bounding box encoding order.

    TensorFlow works with the (y_min, x_min, y_max, x_max) order while we work
    with the (x_min, y_min, x_max, y_min).

    While both encoding options have its advantages and disadvantages we
    decided to use the (x_min, y_min, x_max, y_min), forcing use to switch to
    TensorFlow's every time we want to use a std function that handles bounding
    boxes.

    Args:
        bboxes: A Tensor of shape (total_bboxes, 4)

    Returns:
        bboxes: A Tensor of shape (total_bboxes, 4) with the order swaped.
    """
    with tf.name_scope('BoundingBoxTransform/change_order'):
        first_min, second_min, first_max, second_max = tf.unstack(
            bboxes, axis=1
        )
        bboxes = tf.stack(
            [second_min, first_min, second_max, first_max], axis=1
        )
        return bboxes


def encode(bboxes, gt_boxes):
    with tf.name_scope('BoundingBoxTransform/encode'):
        (
            bboxes_width, bboxes_height, bboxes_urx, bboxes_ury
        ) = get_width_upright(bboxes)

        (
            gt_boxes_width, gt_boxes_height, gt_boxes_urx, gt_boxes_ury
        ) = get_width_upright(gt_boxes)

        targets_dx = (gt_boxes_urx - bboxes_urx) / bboxes_width
        targets_dy = (gt_boxes_ury - bboxes_ury) / bboxes_height

        targets_dw = tf.log(gt_boxes_width / bboxes_width)
        targets_dh = tf.log(gt_boxes_height / bboxes_height)

        targets = tf.concat([
            targets_dx, targets_dy, targets_dw, targets_dh
        ], axis=1)

        return targets


def decode(roi, deltas):
    """
    TODO: Docstring plus param name change.
    """
    with tf.name_scope('BoundingBoxTransform/decode'):
        (
            roi_width, roi_height, roi_urx, roi_ury
        ) = get_width_upright(roi)

        dx, dy, dw, dh = tf.split(deltas, 4, axis=1)

        pred_ur_x = dx * roi_width + roi_urx
        pred_ur_y = dy * roi_height + roi_ury
        pred_w = tf.exp(dw) * roi_width
        pred_h = tf.exp(dh) * roi_height

        bbox_x1 = pred_ur_x - 0.5 * pred_w
        bbox_y1 = pred_ur_y - 0.5 * pred_h

        # This -1. extra is different from reference implementation.
        # TODO: What does this do?
        bbox_x2 = pred_ur_x + 0.5 * pred_w - 1.
        bbox_y2 = pred_ur_y + 0.5 * pred_h - 1.

        bboxes = tf.concat([
            bbox_x1, bbox_y1, bbox_x2, bbox_y2
        ], axis=1)

        return bboxes


def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base anchor to be used as reference of generating all anchors.

    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the wanted widths and heights.

    Scales apply to area of object.

    Args:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to area.

    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    real_heights = (anchors[:, 3] - anchors[:, 1]).astype(np.int)
    real_widths = (anchors[:, 2] - anchors[:, 0]).astype(np.int)

    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )

    return anchors


def generate_anchors(feature_map_shape):
    """Generate anchors for an image.

    Using the feature map (the output of the pretrained network for an image)
    and the anchor references (generated using the specified anchor sizes and
    ratios), we generate a list of anchors.

    Anchors are just fixed bounding boxes of different ratios and sizes that
    are uniformly generated throught the image.

    Args:
        feature_map_shape: Shape of the convolutional feature map used as
            input for the RPN. Should be (batch, height, width, depth).

    Returns:
        all_anchors: A flattened Tensor with all the anchors of shape
            `(num_anchors_per_points * feature_width * feature_height, 4)`
            using the (x1, y1, x2, y2) convention.
    """

    anchor_reference = generate_anchors_reference(
        ANCHOR_BASE_SIZE, ANCHOR_RATIOS, ANCHOR_SCALES
    )
    with tf.variable_scope('generate_anchors'):
        grid_width = feature_map_shape[2]  # width
        grid_height = feature_map_shape[1]  # height
        shift_x = tf.range(grid_width) * OUTPUT_STRIDE
        shift_y = tf.range(grid_height) * OUTPUT_STRIDE
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack(
            [shift_x, shift_y, shift_x, shift_y],
            axis=0
        )

        shifts = tf.transpose(shifts)
        # Shifts now is a (H x W, 4) Tensor

        # Expand dims to use broadcasting sum.
        all_anchors = (
            np.expand_dims(anchor_reference, axis=0) +
            tf.expand_dims(shifts, axis=1)
        )

        # Flatten
        all_anchors = tf.reshape(
            all_anchors, (-1, 4)
        )

    return all_anchors


def filter_proposals(proposals, scores):
    """Filters zero-area proposals."""

    (x_min, y_min, x_max, y_max) = tf.unstack(proposals, axis=1)
    zero_area_filter = tf.greater(
        tf.maximum(x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0),
        0.0
    )
    proposal_filter = zero_area_filter

    scores = tf.boolean_mask(
        scores, proposal_filter,
        name='filtered_scores'
    )
    proposals = tf.boolean_mask(
        proposals, proposal_filter,
        name='filtered_proposals'
    )

    return proposals, scores


def apply_nms(proposals, scores):
    """Applies non-maximum suppression to proposals."""
    # Get top `pre_nms_top_n` indices by sorting the proposals by score.
    # Sorting is done to reduce the number of proposals to run NMS in, and also
    # to return proposals ordered by score.
    k = tf.minimum(PRE_NMS_TOP_N, tf.shape(scores)[0])
    top_k = tf.nn.top_k(scores, k=k)

    sorted_top_proposals = tf.gather(proposals, top_k.indices)
    sorted_top_scores = top_k.values

    with tf.name_scope('nms'):
        # We reorder the proposals into TensorFlows bounding box order
        # for `tf.image.non_max_suppression` compatibility.
        proposals_tf_order = change_order(sorted_top_proposals)
        # We cut the pre_nms filter in pure TF version and go straight
        # into NMS.
        selected_indices = tf.image.non_max_suppression(
            proposals_tf_order, tf.reshape(
                sorted_top_scores, [-1]
            ),
            POST_NMS_TOP_N, iou_threshold=NMS_THRESHOLD
        )

        # Selected_indices is a smaller tensor, we need to extract the
        # proposals and scores using it.
        nms_proposals_tf_order = tf.gather(
            proposals_tf_order, selected_indices,
            name='gather_nms_proposals'
        )

        # We switch back again to the regular bbox encoding.
        proposals = change_order(nms_proposals_tf_order)
        scores = tf.gather(
            sorted_top_scores, selected_indices,
            name='gather_nms_proposals_scores'
        )

    return proposals, scores


def rcnn_proposals(proposals, bbox_pred, cls_prob, im_shape, num_classes,
                   min_prob_threshold=0.0, class_max_detections=100):
    """
    Args:
        proposals: Tensor with the RPN proposals bounding boxes.
            Shape (num_proposals, 4). Where num_proposals is less than
            POST_NMS_TOP_N (We don't know exactly beforehand)
        bbox_pred: Tensor with the RCNN delta predictions for each proposal
            for each class. Shape (num_proposals, 4 * num_classes)
        cls_prob: A softmax probability for each proposal where the idx = 0
            is the background class (which we should ignore).
            Shape (num_proposals, num_classes + 1)

    Returns:
        objects:
            Shape (final_num_proposals, 4)
            Where final_num_proposals is unknown before-hand (it depends on
            NMS). The 4-length Tensor for each corresponds to:
            (x_min, y_min, x_max, y_max).
        objects_label:
            Shape (final_num_proposals,)
        objects_label_prob:
            Shape (final_num_proposals,)

    """
    selected_boxes = []
    selected_probs = []
    selected_labels = []

    TARGET_VARIANCES = np.array([0.1, 0.1, 0.2, 0.2])

    # For each class, take the proposals with the class-specific
    # predictions (class scores and bbox regression) and filter accordingly
    # (valid area, min probability score and NMS).
    for class_id in range(num_classes):
        # Apply the class-specific transformations to the proposals to
        # obtain the current class' prediction.
        class_prob = cls_prob[:, class_id + 1]  # 0 is background class.
        class_bboxes = bbox_pred[:, (4 * class_id):(4 * class_id + 4)]
        raw_class_objects = decode(
            proposals,
            class_bboxes * TARGET_VARIANCES,
        )

        # Clip bboxes so they don't go out of the image.
        class_objects = clip_boxes(raw_class_objects, im_shape)

        # Filter objects based on the min probability threshold and on them
        # having a valid area.
        prob_filter = tf.greater_equal(class_prob, min_prob_threshold)

        (x_min, y_min, x_max, y_max) = tf.unstack(class_objects, axis=1)
        area_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0)
            * tf.maximum(y_max - y_min, 0.0),
            0.0
        )

        object_filter = tf.logical_and(area_filter, prob_filter)

        class_objects = tf.boolean_mask(class_objects, object_filter)
        class_prob = tf.boolean_mask(class_prob, object_filter)

        # We have to use the TensorFlow's bounding box convention to use
        # the included function for NMS.
        class_objects_tf = change_order(class_objects)

        # Apply class NMS.
        class_selected_idx = tf.image.non_max_suppression(
            class_objects_tf, class_prob, class_max_detections,
            iou_threshold=CLASS_NMS_THRESHOLD
        )

        # Using NMS resulting indices, gather values from Tensors.
        class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
        class_prob = tf.gather(class_prob, class_selected_idx)

        # Revert to our bbox convention.
        class_objects = change_order(class_objects_tf)

        # We append values to a regular list which will later be
        # transformed to a proper Tensor.
        selected_boxes.append(class_objects)
        selected_probs.append(class_prob)
        # In the case of the class_id, since it is a loop on classes, we
        # already have a fixed class_id. We use `tf.tile` to create that
        # Tensor with the total number of indices returned by the NMS.
        selected_labels.append(
            tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
        )

    # We use concat (axis=0) to generate a Tensor where the rows are
    # stacked on top of each other
    objects = tf.concat(selected_boxes, axis=0)
    proposal_label = tf.concat(selected_labels, axis=0)
    proposal_label_prob = tf.concat(selected_probs, axis=0)

    # Get top-k detections of all classes.
    k = tf.minimum(
        TOTAL_MAX_DETECTIONS,
        tf.shape(proposal_label_prob)[0]
    )

    top_k = tf.nn.top_k(proposal_label_prob, k=k)
    top_k_proposal_label_prob = top_k.values
    top_k_objects = tf.gather(objects, top_k.indices)
    top_k_proposal_label = tf.gather(proposal_label, top_k.indices)

    return (
        top_k_objects,
        top_k_proposal_label,
        top_k_proposal_label_prob,
    )


def build_base_network(inputs):
    """Obtain the feature map for an input image."""
    # TODO: Not "building" anymore, change name.
    # Pre-process inputs as required by the Resnet (just substracting means).
    means = tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], dtype=tf.float32)
    processed_inputs = inputs - means

    _, endpoints = resnet_v1_101(
        processed_inputs,
        training=False,
        global_pool=False,
        output_stride=OUTPUT_STRIDE,
    )

    feature_map = endpoints['resnet_v1_101/block3']

    return feature_map


def build_rpn(feature_map):
    rpn_conv = tf.layers.conv2d(
        feature_map,
        filters=512,
        kernel_size=[3, 3],
        activation=tf.nn.relu6,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
        padding='same',
        name='rpn/conv',
    )

    num_anchors = len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)
    rpn_cls = tf.layers.conv2d(
        rpn_conv,
        filters=num_anchors * 2,
        kernel_size=[1, 1],
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
        name='rpn/cls_conv',
    )
    rpn_cls_score = tf.reshape(rpn_cls, [-1, 2])
    rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

    rpn_bbox = tf.layers.conv2d(
        rpn_conv,
        filters=num_anchors * 4,
        kernel_size=[1, 1],
        kernel_initializer=(
            tf.random_normal_initializer(mean=0.0, stddev=0.001)
        ),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
        name='rpn/bbox_conv',
    )
    rpn_bbox_pred = tf.reshape(rpn_bbox, [-1, 4])

    return rpn_bbox_pred, rpn_cls_prob


def build_rcnn(features, num_classes):
    rcnn_cls_score = tf.layers.dense(
        features,
        num_classes + 1,
        name='rcnn/fc_classifier',
    )

    rcnn_cls_prob = tf.nn.softmax(rcnn_cls_score)

    rcnn_bbox = tf.layers.dense(
        features,
        num_classes * 4,
        name='rcnn/fc_bbox',
    )

    return rcnn_bbox, rcnn_cls_prob


def build_model(inputs, num_classes, min_prob_threshold=0.0):
    """Build the whole model.

    Args:
        inputs: A Tensor holding the image, of shape `(1, height, width, 3)`.

    Returns:

    """
    # Build the base network.
    feature_map = build_base_network(inputs)

    # Build the RPN.
    rpn_bbox_pred, rpn_cls_prob = build_rpn(feature_map)

    # Generate proposals from the RPN's output by decoding the bounding boxes
    # according to the configured anchors.
    anchors = generate_anchors(tf.shape(feature_map))
    proposals = decode(anchors, rpn_bbox_pred)

    # Get the (positive-object) scores from the RPN.
    # TODO: Could be just one dimension, right? (If not for compatibility.)
    scores = tf.reshape(rpn_cls_prob[:, 1], [-1])

    # Filter proposals with negative areas.
    proposals, scores = filter_proposals(proposals, scores)

    # Apply non-maximum suppression to proposals.
    proposals, scores = apply_nms(proposals, scores)

    # RoI pooling.
    im_shape = inputs.shape[1:3]
    pooled = roi_pooling(feature_map, proposals, im_shape)

    # Apply the tail of the base network.
    features = resnet_v1_101_tail(pooled)[0]
    features = tf.reduce_mean(features, [1, 2])

    # Build the R-CNN.
    bbox_pred, cls_prob = build_rcnn(features, num_classes)

    # Proposals.
    # TODO: Not working when running in script (above?).
    objects, labels, probs = rcnn_proposals(
        proposals, bbox_pred, cls_prob, im_shape, num_classes,
        min_prob_threshold=min_prob_threshold,
    )

    return objects, labels, probs


def normalize_bboxes(proposals, im_shape):
    """
    Gets normalized coordinates for RoIs (between 0 and 1 for cropping)
    in TensorFlow's order (y1, x1, y2, x2).

    Args:
        roi_proposals: A Tensor with the bounding boxes of shape
            (total_proposals, 4), where the values for each proposal are
            (x_min, y_min, x_max, y_max).
        im_shape: A Tensor with the shape of the image (height, width).

    Returns:
        bboxes: A Tensor with normalized bounding boxes in TensorFlow's
            format order. Its should is (total_proposals, 4).
    """
    with tf.name_scope('normalize_bboxes'):
        im_shape = tf.cast(im_shape, tf.float32)

        x1, y1, x2, y2 = tf.unstack(
            proposals, axis=1
        )

        x1 = x1 / im_shape[1]
        y1 = y1 / im_shape[0]
        x2 = x2 / im_shape[1]
        y2 = y2 / im_shape[0]

        bboxes = tf.stack([y1, x1, y2, x2], axis=1)

        return bboxes


def roi_pooling(feature_map, proposals, im_shape):
    """Perform RoI pooling.

    This is a simplified method than what's done in the paper that obtains
    similar results. We crop the proposal over the feature map and resize it
    bilinearly.
    """
    ROI_POOL_WIDTH = 7
    ROI_POOL_HEIGHT = 7

    # Get normalized bounding boxes.
    bboxes = normalize_bboxes(proposals, im_shape)
    bboxes_shape = tf.shape(bboxes)

    # Generate fake batch ids: since we're using a batch size of one, all the
    # `ids` for the bounding boxes are zero.
    batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)

    # Apply crop and resize with extracting a crop double the desired size.
    crops = tf.image.crop_and_resize(
        feature_map, bboxes, batch_ids,
        [ROI_POOL_WIDTH * 2, ROI_POOL_HEIGHT * 2], name='crops'
    )

    # Applies max pool with [2,2] kernel to reduce the crops to half the
    # size, and thus having the desired output.
    pooled = tf.nn.max_pool(
        crops, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'
    )

    return pooled


@click.command()
@click.argument('image-path')
def main(image_path):
    raw_image = Image.open(image_path)
    image = np.expand_dims(raw_image.convert('RGB'), axis=0)

    NUM_CLASSES = 80

    tf.enable_eager_execution()
    with tfe.restore_variables_on_create('checkpoint/fasterrcnn'):
        objects, labels, probs = build_model(image, NUM_CLASSES)

    draw_bboxes(raw_image, objects[:10])
    raw_image.save('out.png')


if __name__ == '__main__':
    main()
