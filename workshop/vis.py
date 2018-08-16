import math
import matplotlib.pyplot as plt

from PIL import ImageDraw, ImageFont

from workshop.image import to_image


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


def image_grid(count, columns=4, sizes=(5, 3)):
    rows = math.ceil(count / columns)

    width, height = sizes

    figsize = (columns * width, rows * height)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)

    # Default configuration for each axis.
    for ax in axes.ravel():
        ax.axis('off')

    return axes.ravel()
