import click
import json
import numpy as np
import os
import sys

from PIL import Image, ImageDraw, ImageFont

from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.predicting import PredictorNetwork


# Goes to `luminoth.<something>`. (Place where all generic task models are
# located.)
# TODO: `predict_image` creates and runs a Tensorflow session inside. How to
# make compatible with eager? Can we detect wheter there's an active session
# and reuse it? Can we use `tf.eager_enabled`? (Even though we don't support it
# yet.)
# TODO: Can we use `Detector` inside a notebook even if it creates a session
# internally?
class Detector(object):
    """Encapsulates an object detection model behavior.

    In order to perform object detection with a model implemented within
    Luminoth, this class should be used. It provides a common API that
    abstracts away the inner workings of each particular object detection
    model.

    Attributes:
        classes (list of str): Ordered class names for the detector.
        prob (float): Default probability threshold for predictions.
    """

    DEFAULT_CHECKPOINT = 'accurate'

    def __init__(self, checkpoint=None, config=None, prob=0.7, classes=None):
        """Instantiate a detector object with the appropriate config.

        Arguments:
            checkpoint (str): Checkpoint name to instantiate the detector as.
            config (dict): Configuration parameters describing the desired
                model. See `get_config` to load a config file.

        Note:
            Only one of the parameters must be specified. If none is, we
            default to loading the checkpoint indicated by
            `DEFAULT_CHECKPOINT`.
        """
        if checkpoint is not None and config is not None:
            raise ValueError(
                'Only one of `checkpoint` or `config` must be specified in '
                'order to instantiate a Detector.'
            )

        if checkpoint is None and config is None:
            # Neither checkpoint no config specified, default to
            # `DEFAULT_CHECKPOINT`.
            checkpoint = self.DEFAULT_CHECKPOINT

        if checkpoint:
            config = get_checkpoint_config(checkpoint)

        # Prevent the model itself from filtering its proposals (default
        # value of 0.5 is in use in the configs).
        # TODO: A model should always return all of its predictions. The
        # filtering should be done (if at all) by PredictorNetwork.
        if config.model.type == 'fasterrcnn':
            config.model.rcnn.proposals.min_prob_threshold = 0.0
        elif config.model.type == 'ssd':
            config.model.proposals.min_prob_threshold = 0.0

        # TODO: Remove dependency on `PredictorNetwork` or clearly separate
        # responsibilities.
        self._network = PredictorNetwork(config)

        self.prob = prob

        # TODO: What to do when it's not present?
        self._model_classes = self._network.class_labels
        if classes:
            self.classes = set(classes)
            if not set(self._model_classes).issuperset(self.classes):
                raise ValueError(
                    '`classes` must be contained in the detector\'s classes. '
                    'Available classes are: {}.'.format(self._model_classes)
                )
        else:
            self.classes = set(self._model_classes)

    def predict(self, images, prob=None, classes=None):
        """Run the detector through a set of images.

        Arguments:
            images (numpy.ndarray or list): Either array of dimensions
                `(height, width, channels)` (single image) or array of
                dimensions `(number_of_images, height, width, channels)`
                (multiple images). If a list, must be a list of rank 3 arrays.
            prob (float): Override configured probability threshold for
                predictions.
            classes (set of str): Override configured class names to consider.

        Returns:
            Either list of objects detected in the image (single image case) or
            list of list of objects detected (multiple images case).

            In the multiple images case, the outer list has `number_of_images`
            elements, while the inner ones have the number of objects detected
            in each image.

            Each object has the format::

                {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'label': '<cat|dog|person|...>' | 0..C,
                    'prob': prob
                }

            The coordinates are integers, where `(x_min, y_min)` are the
            coordinates of the top-left corner of the bounding box, while
            `(x_max, y_max)` the bottom-right. By convention, the top-left
            corner of the image is coordinate `(0, 0)`.

            The probability, `prob`, is a float between 0 and 1, indicating the
            confidence of the detection being correct.

            The label of the object, `label`, may be either a string if the
            classes file for the model is found or an integer between 0 and the
            number of classes `C`.

        """
        # If it's a single image (ndarray of rank 3), turn into a list.
        single_image = False
        if not isinstance(images, list):
            if len(images.shape) == 3:
                images = [images]
                single_image = True

        if prob is None:
            prob = self.prob

        if classes is None:
            classes = self.classes
        else:
            classes = set(classes)

        # TODO: Remove the loop. Neither Faster R-CNN nor SSD support batch
        # size yet, so it's the same for now.
        predictions = []
        for image in images:
            predictions.append([
                pred for pred in self._network.predict_image(image)
                if pred['prob'] >= prob and pred['label'] in classes
            ])

        if single_image:
            predictions = predictions[0]

        return predictions

    # TODO: Change name. Should also return something else; if we're getting
    # embeddings, we probably want everything to be more efficient (e.g. no
    # dicts).
    # TODO: Implement.
    def predict_with_embeddings(images):
        raise NotImplementedError


# Goes to `luminoth.vis`.
def hex_to_rgb(x):
    return tuple([int(x[i:i + 2], 16) for i in (0, 2, 4)])


# Goes to `luminoth.vis`.
def build_colormap():
    """Builds a colormap function that maps labels to colors.

    Returns:
        Function that receives a label and returns a color tuple `(R, G, B)`
        for said label.
    """
    # Build the 10-color palette to be used for all classes.
    palette = (
        '1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf'
    )
    colors = [hex_to_rgb(palette[i:i + 6]) for i in range(0, len(palette), 6)]

    seen_labels = {}

    def colormap(label):
        # If label not yet seen, get the next value in the palette sequence.
        if label not in seen_labels:
            seen_labels[label] = colors[len(seen_labels) % len(colors)]

        return seen_labels[label]

    return colormap


# Goes to `luminoth.io`.
def read_image(path):
    """Reads an image located at `path` into an array.

    Arguments:
        path (str): Path to a valid image file in the filesystem.

    Returns:
        `numpy.ndarray` of size `(height, width, channels)`.
    """
    full_path = os.path.expanduser(path)
    return np.array(Image.open(full_path).convert('RGB'))


# Goes to `luminoth.vis`.
def draw_rectangle(draw, coordinates, color, width=1):
    """Draw a rectangle with an optional width."""
    # Add alphas to the color so we have a small overlay over the object.
    fill = color + (30,)
    outline = color + (255,)

    # Pillow doesn't support width in rectangles, so we must emulate it with a
    # loop.
    for i in range(width):
        coords = [
            coordinates[0] - i,
            coordinates[1] - i,
            coordinates[2] + i,
            coordinates[3] + i,
        ]

        # Fill must be drawn only for the first rectangle, or the alphas will
        # add up.
        if i == 0:
            draw.rectangle(coords, fill=fill, outline=outline)
        else:
            draw.rectangle(coords, outline=outline)


# Goes to `luminoth.vis`.
def get_font():
    if sys.platform == 'win32':
        font_names = ['Arial']
    elif sys.platform in ['linux', 'linux2']:
        font_names = ['DejaVuSans-Bold', 'DroidSans-Bold']
    elif sys.platform == 'darwin':
        font_names = ['Menlo', 'Helvetica']

    font = None
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name)
            break
        except IOError:
            continue

    return font


# Goes to `luminoth.vis`.
SYSTEM_FONT = get_font()


# Goes to `luminoth.vis`.
def draw_label(draw, coords, color, label, prob):
    """Draw a box with the label and probability."""
    # Attempt to get a native TTF font. If not, use the default bitmap font.
    global SYSTEM_FONT
    if SYSTEM_FONT:
        label_font = SYSTEM_FONT.font_variant(size=16)
        prob_font = SYSTEM_FONT.font_variant(size=12)
    else:
        label_font = ImageFont.load_default()
        prob_font = ImageFont.load_default()

    label = str(label)  # `label` may not be a string.
    prob = '({:.2f})'.format(prob)  # Turn `prob` into a string.

    # We want the probability font to be smaller, so we'll write the label in
    # two steps.
    label_w, label_h = label_font.getsize(label)
    prob_w, prob_h = prob_font.getsize(prob)

    # Get margins to manually adjust the spacing. The margin goes between each
    # segment (i.e. margin, label, margin, prob, margin).
    margin_w, margin_h = label_font.getsize('M')
    margin_w *= 0.2
    _, full_line_height = label_font.getsize('Mq')

    # Draw the background first, considering all margins and the full line
    # height.
    background_coords = [
        coords[0],
        coords[1],
        coords[0] + label_w + prob_w + 3 * margin_w,
        coords[1] + full_line_height * 1.15,
    ]
    draw.rectangle(background_coords, fill=color + (255,))

    # Then write the two pieces of text.
    draw.text([
        coords[0] + margin_w,
        coords[1],
    ], label, font=label_font)

    draw.text([
        coords[0] + label_w + 2 * margin_w,
        coords[1] + (margin_h - prob_h),
    ], prob, font=prob_font)


# Goes to `luminoth.vis`.
def vis_objects(image, objects, colormap=None, labels=True):
    """Visualize objects as returned by `Detector`.

    Arguments:
        image (numpy.ndarray): Image to draw the bounding boxes on.
        objects (list of dicts or dict): List of objects as returned by a
            `Detector` instance.
        colormap (function): Colormap function to use for the objects.
        labels (boolean): Whether to draw labels.

    Returns:
        A PIL image with the detected objects' bounding boxes and labels drawn.
        Can be casted to a `numpy.ndarray` by using `numpy.array` on the
        returned object.
    """
    if not isinstance(objects, list):
        objects = [objects]

    if colormap is None:
        colormap = build_colormap()

    image = Image.fromarray(image.astype(np.uint8))

    draw = ImageDraw.Draw(image, 'RGBA')
    for obj in objects:
        # TODO: Can we do image resolution-agnostic?
        color = colormap(obj['label'])
        draw_rectangle(draw, obj['bbox'], color, width=3)
        if labels:
            draw_label(draw, obj['bbox'][:2], color, obj['label'], obj['prob'])

    return image


@click.command()
@click.argument('image-path')
@click.option('--checkpoint', default='accurate')
@click.option('--save-path', default='out.png')
def main(image_path, checkpoint, save_path):
    image = read_image(image_path)

    detector = Detector(checkpoint)
    objects = detector.predict(image)

    click.echo(json.dumps(objects, indent=2))

    vis_objects(
        image,
        objects
    ).save(save_path)


if __name__ == '__main__':
    main()
