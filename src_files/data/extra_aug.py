import mmcv
import numpy as np
from numpy import random
from PIL import Image

class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img):
        # random brightness
        img = np.array(img, dtype=np.float32)
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class RandomCrop(object):

    def __init__(self, min_crop_size=0.8):
        # 1: return ori img
        self.min_crop_size = min_crop_size

    def __call__(self, img):
        h, w, c = img.shape

        for i in range(50):
            new_w = random.uniform(self.min_crop_size * w, w)
            new_h = random.uniform(self.min_crop_size * h, h)

            # h / w in [0.5, 2]
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            left = random.uniform(w - new_w)
            top = random.uniform(h - new_h)

            patch = np.array((int(left), int(top), int(left + new_w),
                              int(top + new_h)))

            img = img[patch[1]:patch[3], patch[0]:patch[2]]

            return img
        return img
