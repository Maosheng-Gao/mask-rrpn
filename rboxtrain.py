#!/usr/bin/env python3
import sys
import json, os
import skimage.draw
from maskrcnn import utils
import numpy as np
import cv2
from maskrcnn import training, dataset, config


class TrainingConfig(config.Config):
    NAME = "Fiber"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    IMAGE_MAX_DIM = 448
    BACKBONE = "resnet50"
    regressor = None
    USE_MINI_MASK = False


class FiberDataset(utils.Dataset):

    def load_data(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("fiber", 1, "fiber")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #         #   'regions': {
        #         #       '0': {
        #         #           'region_attributes': {},
        #         #           'shape_attributes': {
        #         #               'all_points_x': [...],
        #         #               'all_points_y': [...],
        #         #               'name': 'polygon'}},
        #         #       ... more regions ...
        #         #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "fiber",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "fiber":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = cv2.imread(self.image_info[image_id]['path'])
        # print(self.image_info[image_id]['path'])
        # edges = cv2.Canny(image, 50, 100, apertureSize=3)  # apertureSize参数默认其实就是3
        # thresh = 255 - cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 89, 4)
        # image = cv2.merge([image, edges, thresh])
        return image


if __name__ == '__main__':
    dataset_dir = './dataset/'

    model = "last"
    dataset_base = "./dataset/"

    dataset_train = FiberDataset()
    dataset_train.load_data(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_valid = FiberDataset()
    dataset_valid.load_data(dataset_dir, "val")
    dataset_valid.prepare()

    config = TrainingConfig()
    config.regressor = 'deltas'
    if config.regressor:
        config.NAME="{0}_{1}_{2}".format(config.NAME, config.BACKBONE, config.regressor)
        print("Configuration Name: ", config.NAME)

    net = training.Training(config)

    if model:
        if model.lower() == "last":
            model_path = net.find_last()[1]
        else:
            model_path = model
        print("Model: {0}".format(model_path))
        net.load_weights(model_path, by_name=True)

        print("Training network heads")
        net.train(
            dataset_train,
            dataset_valid,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')

        print("Fine tune Resnet stage 4 and up")
        net.train(
            dataset_train,
            dataset_valid,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='4+')

        print("Fine tune all layers")
        net.train(
            dataset_train,
            dataset_valid,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=160,
            layers='all')
