# Luminoth Guided Demo — PyImageConf 2018

![Luminoth Logo](https://user-images.githubusercontent.com/270983/31414425-c12314d2-ae15-11e7-8cc9-42d330b03310.png)

## Introduction

In this part of the workshop, we will use the [Luminoth](https://luminoth.ai/) toolkit, which has an implementation of the Faster R-CNN method we have just seen.

The idea is to learn the usage, and be prepared to solve real world object detection problems with the tool.

### Installation and setting up

To use Luminoth, **TensorFlow** must be installed beforehand.

If you want **GPU support**, which is very desirable given that we are going to run compute-intensive tasks, you should install the GPU version of TensorFlow first.

We are going to start by creating and activating a **Virtualenv** for the work that follows:

    python -m venv .virtualenvs/luminoth
    source .virtualenvs/luminoth/bin/activate

Now, we just need to install TensorFlow (GPU version) and Luminoth:

    pip install tensorflow-gpu
    pip install luminoth

To check that everything works as expected, you should run `lumi --help` and get something like this:

    Usage: lumi [OPTIONS] COMMAND [ARGS]...

    Options:
      -h, --help  Show this message and exit.

    Commands:
      checkpoint  Groups of commands to manage checkpoints
      cloud       Groups of commands to train models in the...
      dataset     Groups of commands to manage datasets
      eval        Evaluate trained (or training) models
      predict     Obtain a model's predictions.
      server      Groups of commands to serve models
      train       Train models

Congratulations! Now Luminoth is setup and you can start playing around.

## Using Luminoth 101

In this section, we are going to walk through the baby steps of using Luminoth.

The first thing is being familiarized with the Luminoth CLI tool, that is, the tool that you interacted with using the  `lumi` command. This tool is the main way to interact with Luminoth, allowing you to train new models, evaluate them, use them for predictions, manage your checkpoints and more.

To start, you should fetch a couple of images/videos from the internet. We will try to play around with traffic-related stuff (cars, pedestrians, bicycles, etc), so we want images that relate to what you would see on the street. To make it easier, we have fetched a couple for you in S3 here:

    s3://pyimageconf-2018-obj-det-workshop/test-media/

You may download those directly with to your directory of choice using [AWS CLI](https://aws.amazon.com/cli/):

    aws --no-sign-request s3 cp --recursive s3://pyimageconf-2018-obj-det-workshop/test-media <your-directory>

But you can Google and try it out with your own images!

### Using from the shell: detecting objects in an image or video

Fire up the shell and go to the directory where your images are located. Let’s say we want Luminoth to predict the objects present in one of these pictures (`bicycling-1160860_1280.jpg`). The way to do that is by running the following command:

    lumi predict bicycling-1160860_1280.jpg

You will see the following output:

    Found 1 files to predict.
    Neither checkpoint not config specified, assuming `accurate`.
    Checkpoint not found. Check remote repository? [y/N]:

What happens is that you didn’t tell Luminoth what an “object” is for you, nor have taught it how to recognize said objects.

One way to do this is to use a **pre-trained model** that has been trained to detect popular types of objects. For example, it can be a model trained with [COCO dataset](http://cocodataset.org/) or [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/). Moreover, each pre-trained model might be associated with a different algorithm. This is what **checkpoints** are: they correspond to the weights of a particular model (Faster R-CNN or SSD), trained with a particular dataset.

The case of “accurate” is just a label for a particular Deep Learning model underneath, in this case, Faster R-CNN, trained with images from the COCO dataset. The idea is that Luminoth assumes that by default you want the most accurate predictions, and it will use the most accurate model that it knows about. At this time, it is Faster R-CNN, but that could be replaced in the future and you, as a user, wouldn’t have to change your code.

Type ‘y’ and Luminoth will check the remote index, to see what checkpoints are available. Luminoth currently hosts pre-trained checkpoints for Faster R-CNN (COCO) and SSD (Pascal VOC), though more will be added.

Type ‘y’ again after it prompts you to download the checkpoint. The checkpoints will be stored in `~/.luminoth` folder.

After the download finishes, you will get the predictions for your image in JSON format in the standard output:

    Predicting bicycling-1160860_1280.jpg... done.
    {"file": "bicycling-1160860_1280.jpg", "objects": [{"bbox": [393, 300, 631, 748], "label": "person", "prob": 0.9996}, {"bbox": [978, 403, 1074, 608], "label": "person", "prob": 0.9965}, {"bbox": [670, 382, 775, 596], "label": "person", "prob": 0.9949}, {"bbox": [746, 421, 877, 743], "label": "person", "prob": 0.9947}, {"bbox": [431, 517, 575, 776], "label": "bicycle", "prob": 0.9876}, {"bbox": [775, 561, 860, 792], "label": "bicycle", "prob": 0.9775}, {"bbox": [986, 499, 1057, 636], "label": "bicycle", "prob": 0.9547}, {"bbox": [1135, 420, 1148, 451], "label": "person", "prob": 0.8286}, {"bbox": [683, 480, 756, 621], "label": "bicycle", "prob": 0.7845}, {"bbox": [772, 394, 853, 478], "label": "person", "prob": 0.6044}, {"bbox": [384, 318, 424, 365], "label": "baseball glove", "prob": 0.6037}, {"bbox": [700, 412, 756, 471], "label": "backpack", "prob": 0.5078}, {"bbox": [606, 311, 637, 353], "label": "baseball glove", "prob": 0.5066}]}

This is probably unintelligible to you, and also not apt for machine consumption since it’s mixed with other things in the standard output. However, it's also possible to get the JSON file with the objects plus the actual image with the overlayed bounding boxes. With these commands we can output everything to a `preds` directory:


    mkdir preds
    lumi predict bicycling-1160860_1280.jpg -f preds/objects.json -d preds/

If you fetch the resulting image, it should look like this:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_33F2D71A4873F6AC0B5A91D5E7CF81880C953D375093FC504D79B37E7275A897_1534544332622_pred_bicycling-1160860_1280.png)


Not bad!

You can also run predictions on a **video file** in the exact same way. Note that this is basically independent frame by frame predictions, and has no tracking or interpolation. Try it out! Depending on the length of the video, it can take a while :)

### Exploring pre-trained checkpoints

Whenever you wish to work with checkpoints, you must first run the `lumi checkpoint refresh` command, so Luminoth knows about the checkpoints that it has available for download. The remote index can be updated periodically.

After refreshing the local index, you can list the available checkpoints running `lumi checkpoint list`:


    ================================================================================
    |           id |                  name |       alias | source |         status |
    ================================================================================
    | e1c2565b51e9 |   Faster R-CNN w/COCO |    accurate | remote |     DOWNLOADED |
    | aad6912e94d9 |      SSD w/Pascal VOC |        fast | remote | NOT_DOWNLOADED |
    ================================================================================

Here, you can see the “accurate” checkpoint that we have used for our predictions before, and that we also have another “fast” checkpoint that is the SSD model trained with Pascal VOC dataset. Let’s get some information about the “accurate” checkpoint:


    Faster R-CNN w/COCO (e1c2565b51e9, accurate)
    Base Faster R-CNN model trained with the full COCO dataset.

    Model used: fasterrcnn
    Dataset information
        Name: COCO
        Number of classes: 80

    Creation date: 2018-04-17T16:58:00.658815
    Luminoth version: v0.1.1

    Source: remote (DOWNLOADED)
    URL: https://github.com/tryolabs/luminoth/releases/download/v0.1.0/e1c2565b51e9.tar

You can see that this dataset consists of 80 classes, and other useful information. Let’s see what the `fast` checkpoint is about:


    SSD w/Pascal VOC (aad6912e94d9, fast)
    Base SSD model trained with the full Pascal dataset.

    Model used: ssd
    Dataset information
        Name: Pascal VOC
        Number of classes: 20

    Creation date: 2018-04-12T17:42:01.598779
    Luminoth version: v0.1.1

    Source: remote (NOT_DOWNLOADED)
    URL: https://github.com/tryolabs/luminoth/releases/download/v0.1.0/aad6912e94d9.tar

If you want to get predictions for an image or video using a specific checkpoint (for example, `fast`) you can do so by using the `--checkpoint` parameter:

    lumi predict bicycling-1160860_1280.jpg --checkpoint fast -f preds/objects.json -d preds/

Inspecting the image, you’ll see that it doesn't work as nicely as the `accurate` checkpoint.

Also note that in every command where we used the alias of checkpoint, we could also have used the id.


### The built-in interface for playing around

Luminoth also includes a simple web frontend so you can play around with detected objects in images using different thresholds.

To launch this, simply type `lumi server web` and then open your browser at http://localhost:5000.
If you are running on an external VM, you can do `lumi server web --host 0.0.0.0 --port <port>` to open in a custom port.

Now, select an image and submit! See the results.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_33F2D71A4873F6AC0B5A91D5E7CF81880C953D375093FC504D79B37E7275A897_1534547066842_file.png)

You can go ahead and change the “probability threshold” slidebar and see how the detection looks with more or less filtering. You’ll see that as you lower the threshold, more objects appear (and many times these are incorrect), while increasing the threshold makes the most accurate guesses but misses many of the objects you wish to detect.

## Using your own dataset

Even though pre-trained checkpoints are really useful, most of the time you will want to train an object detector using your own dataset. For this, you need a source of images and their corresponding bounding box coordinates and labels, in some format that Luminoth can understand.

Luminoth reads datasets natively only in TensorFlow’s [TFRecords format](https://www.tensorflow.org/guide/datasets#consuming_tfrecord_data). This is a binary format that will let Luminoth consume the data very efficiently. In order to use a custom dataset, you must first transform whatever format your data is in, to TFRecords files (one for each split — train, val, test). Fortunately, Luminoth provides several [CLI tools](https://luminoth.readthedocs.io/en/latest/usage/dataset.html) for transforming popular dataset format (such as Pascal VOC, ImageNet, COCO, CSV, etc.) into TFrecords.

### Building a custom traffic dataset using OpenImages

[OpenImages V4](https://storage.googleapis.com/openimages/web/index.html) is the largest existing dataset with object location annotations. It contains 15.4M bounding-boxes for 600 categories on 1.9M images, making it a very good choice for getting example images of a variety of (not niche-domain) classes (persons, cars, dolphin, blender, etc).

Normally, we would start downloading [the annotation files](https://storage.googleapis.com/openimages/web/download.html) ([this](https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv) and [this](https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels-boxable.csv), for train) and the [class description](https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv) file. Note that the files with the annotations themselves are pretty large, totalling over 1.5 GB (and this, without downloading a single image!).

This time, we have gone over the classes available in the OpenImages dataset, and created files for a **reduced OpenImages** which only contains some classes pertaining to **traffic**. The following were hand-picked from after examining the full `class-descriptions-boxable.csv` file:

    /m/015qff,Traffic light
    /m/0199g,Bicycle
    /m/01bjv,Bus
    /m/01g317,Person
    /m/04_sv,Motorcycle
    /m/07r04,Truck
    /m/0h2r6,Van
    /m/0k4j,Car

You can download this dataset [HERE](https://s3-us-west-1.amazonaws.com/pyimageconf-2018-obj-det-workshop/openimages-reduced-traffic.tar.gz). Then, extract it to a folder of choice (`tar -xzf openimages-reduced-traffic.tar.gz`).

Luminoth supports a dataset reader that can take OpenImages format. As the dataset is so large, this will never download every image, but fetch only what we want to use and store directly in the TFRecords format.

Go into the folder where you extracted the dataset and run the following command:

    lumi dataset transform \
            --type openimages \
            --data-dir . \
            --output-dir ./out \
            --split train  \
            --class-examples 100 \
            --only-classes=/m/015qff,/m/0199g,/m/01bjv,/m/01g317,/m/04_sv,/m/07r04,/m/0h2r6,/m/0k4j

This will generate TFRecord file for the `train` split. You should get something like this in your terminal after the command finishes:

    INFO:tensorflow:Saved 360 records to "./out/train.tfrecords"
    INFO:tensorflow:Composition per class (train):
    INFO:tensorflow:        Person (/m/01g317): 380
    INFO:tensorflow:        Car (/m/0k4j): 255
    INFO:tensorflow:        Bicycle (/m/0199g): 126
    INFO:tensorflow:        Bus (/m/01bjv): 106
    INFO:tensorflow:        Traffic light (/m/015qff): 105
    INFO:tensorflow:        Truck (/m/07r04): 101
    INFO:tensorflow:        Van (/m/0h2r6): 100
    INFO:tensorflow:        Motorcycle (/m/04_sv): 100

Note that:

- As we are using `--only-classes`, this command will work even if we are using the full annotation files of OpenImages (and not the reduced version we provided, for limiting bandwidth).
- This will download only the images that it needs. It will not store them into any intermediate location, but directly in the TFRecord file.
- We are using `--max-per-class` of 100. This setting will make it stop when every class has at least 100 examples. However, some classes may end up with many more; for example here it needed to get 380 instances of persons to get 100 motorcycles, considering the first 360 images.
- We could also have used `--limit-examples` option so we know the number of records in our final dataset beforehand.

Of course, this dataset is **way too small** for any meaningful training to go on, but we are just showcasing. In real life, you would use a much larger value for `--max-per-class` (ie. 15000) or `--limit-examples`.

## Training models with custom data

Now that we have created our (toy) dataset, we can proceed to train our model.

### The configuration file

Training orchestration, including the model to be used, the dataset location and training schedule, is specified in a YAML config file. This file will be consumed by Luminoth and merged to the default configuration, to start the training session.

You can see a minimal config file example in [sample_config.yml](https://github.com/tryolabs/luminoth/blob/master/examples/sample_config.yml). This file illustrates the entries you’ll most probably need to modify, which are:

- `train.run_name`: the run name for the training session, used to identify it.
- `train.job_dir`: directory in which both model checkpoints and summaries (for TensorBoard consumption) will be saved. The actual files will be stored under `<job_dir>/<run_name>`.
- `dataset.dir`: directory from which to read the TFRecord files.
- `model.type`: model to use for object detection (`fasterrcnn`, or `ssd`).
- `network.num_classes`: number of classes to predict (depends on your dataset).

For looking at all the possible configuration options,  mostly related to the model itself, you can check the [base_config.yml](https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/base_config.yml) file.

### Building the config file for your dataset

Probably the most important setting for training is the **learning rate**. You will most likely want to tune this depending on your dataset, and you can do it via the  `train.learning_rate` setting in the configuration. For example, this would be a good setting for training on the full COCO dataset:

      learning_rate:
        decay_method: piecewise_constant
        boundaries: [250000, 450000, 600000]
        values: [0.0003, 0.0001, 0.00003, 0.00001]

To get to this, you will probably need to run some experiments and see what works best.

    train:
      # Run name for the training session.
      run_name: traffic
      job_dir: <change this directory>
      learning_rate:
        decay_method: piecewise_constant
        # Custom dataset for PyImageConf Workshop
        boundaries: [90000, 160000, 250000]
        values: [0.0003, 0.0001, 0.00003, 0.00001]
    dataset:
      type: object_detection
      dir: <directory with your dataset>
    model:
      type: fasterrcnn
      network:
        num_classes: 8
      anchors:
        # Add one more scale to be better at detecting small objects
        scales: [0.125, 0.25, 0.5, 1, 2]

### Running the training

Assuming you already have both your dataset (TFRecords) and the config file ready, you can start your training session by running the command as follows:

    lumi train -c config.yml

You can use the `-o` option to override any configuration option using dot notation (e.g. `-o model.rpn.proposals.nms_threshold=0.8`).

If you are using a CUDA-based GPU, you can select the GPU to use by setting the `CUDA_VISIBLE_DEVICES` environment variable (see [here](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) for more info).


## Using TensorBoard to visualize the training process

The loss, step.

### Evaluating models

### The mAP metrics

### TensorBoard


## Creating your own checkpoints

https://luminoth.readthedocs.io/en/latest/usage/checkpoints.html

## Using Luminoth from Python code
