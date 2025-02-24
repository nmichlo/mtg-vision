#  MIT License
#
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from tqdm import tqdm
from mtg_vision._old.util import GLOBAL_TIMER
import numpy as np
from math import log
from mtg_vision._old.util.random import GLOBAL_RAN
import os
import cv2


# ========================================================================= #
# Image                                                                     #
# ========================================================================= #


class SampleImg:

    def __init__(self, image, train_sample=True):

        if image.dtype in [np.float64, np.float32]:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.train_sample = train_sample

        if self.train_sample:
            mean = np.mean(image)
            std = np.std(image)
            image = ((image - mean) / (2 * std)) * 128 + 128
            image = np.round(image)
            image = image.astype(np.uint8)
            # leave image as float... otherwise does not correspond with normalisation in detection
            # TODO: why does this not work if I don't convert to uint8?
            (self.integral, self.integral2) = (cv2.integral(image), None)
        else:
            (self.integral, self.integral2) = cv2.integral2(image)

        print(image.shape, self.integral.shape, self.integral2)
        print()

    def __str__(self):
        return str(self.integral)

    # Sum of pixels in region:
    # x element of [x0, x1)
    # y element of [y0, y1)
    def sum(self, x0, y0, x1, y1):
        assert x1 > x0 and y1 > y0
        a = self.integral[y0, x0]
        b = self.integral[y0, x1]
        c = self.integral[y1, x0]
        d = self.integral[y1, x1]
        return d - c - b + a

    def sumNorm(self, x0, y0, x1, y1, mean, std):
        assert x1 > x0 and y1 > y0
        a = self.integral[y0, x0]
        b = self.integral[y0, x1]
        c = self.integral[y1, x0]
        d = self.integral[y1, x1]
        area = (x1-x0)*(y1-y0)
        return (((d-c-b+a) - area*mean)/(2*std) + area) * 128

    # Sum of square of pixels in region:
    # x element of [x0, x1)
    # y element of [y0, y1)
    def sum2(self, x0, y0, x1, y1):
        assert x1 > x0 and y1 > y0
        a = self.integral2[y0, x0]
        b = self.integral2[y0, x1]
        c = self.integral2[y1, x0]
        d = self.integral2[y1, x1]
        return d - c - b + a

    def mean(self, x0, y0, x1, y1):
        assert x1 > x0 and y1 > y0
        sum = self.sum(x0, y0, x1, y1)
        if sum == 0:
            return 0
        area = (x1 - x0) * (y1 - y0)
        return sum / area

    def meanAndStd(self, x0, y0, x1, y1):
        assert x1 > x0 and y1 > y0
        sum = self.sum(x0, y0, x1, y1)
        if sum == 0:
            return (0, 1)
        sum2 = self.sum2(x0, y0, x1, y1)
        area = (x1 - x0) * (y1 - y0)
        mean = sum / area
        std = abs(sum2/area - mean*mean) ** 0.5
        return (mean, std)


    @staticmethod
    def loadImageList(img_list, size=24, cvt_to_sample=False, train_samples=True):
        images = []
        img_folder = os.path.dirname(img_list)
        var_list = [line.strip().split(" ") for line in open(img_list) if line.strip()]
        var_list = [[int(v) if (i > 0) else v for i, v in enumerate(vars)] for vars in var_list]

        for var in var_list:
            img = None
            if len(var) == 6:
                (image_name, c, x, y, w, h) = var
                try:
                    img = cv2.imread(os.path.join(img_folder, image_name), cv2.IMREAD_GRAYSCALE)
                    img = img[y:y+h, x:x+w]
                    img = cv2.resize(img, (size, size))
                except:
                    print("\tError loading: {}!".format(os.path.join(img_folder, image_name)))
            elif len(var) == 1:
                image_name = var[0]
                try:
                    img = cv2.imread(os.path.join(img_folder, image_name), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (size, size))
                except:
                    print("\tError loading: {}!".format(os.path.join(img_folder, image_name)))
            if img is not None:
                images.append(SampleImg(img, train_sample=train_samples) if cvt_to_sample else img)

        print("Loaded and Scaled {} ({}) to {}x{} Images!".format(len(images), len(var_list), size, size))
        return images

    @staticmethod
    def loadImageFolder(img_folder, ext='.pgm', size=24, cvt_to_sample=False, train_samples=True):
        images = []
        img_names = [name for name in os.listdir(img_folder) if os.path.splitext(name)[1].lower() == ext.lower()]
        for img_name in img_names:
            try:
                gray = cv2.imread(os.path.join(img_folder, img_name), cv2.IMREAD_GRAYSCALE)
                gray = cv2.resize(gray, (size, size))
                images.append(SampleImg(gray, train_sample=train_samples) if cvt_to_sample else gray)
            except:
                print("\tError loading: {}!".format(os.path.join(img_folder, img_name)))
        print("Loaded and Scaled {} ({}) to {}x{} Images!".format(len(images), len(img_names), size, size))
        return images


# ========================================================================= #
# Features                                                                  #
# ========================================================================= #


class Feature:

    (HORR_2, VERT_2, HORR_3, VERT_3, CHECKER) = (0, 1, 2, 3, 4)

    TYPES = (HORR_2, VERT_2, HORR_3, VERT_3, CHECKER)

    CELLS = {
        HORR_2: (1, 2),
        VERT_2: (2, 1),
        HORR_3: (1, 3),
        VERT_3: (3, 1),
        CHECKER: (2, 2),
    }

    @staticmethod
    def calculate(sample, x0, y0, x1, y1, feat_type): #About 35% to 45% faster than an iterative method
        data = sample.integral
        if feat_type == Feature.HORR_2:
            p1 = int((x0 + x1)/2)
            a = data[y0, x0]
            b = data[y1, x0]
            c = data[y0, p1]
            d = data[y1, p1]
            e = data[y0, x1]
            f = data[y1, x1]
            return (d-c-b+a) - (f-e-d+c)
        elif feat_type == Feature.VERT_2:
            p1 = int((y0 + y1)/2)
            a = data[y0, x0]
            b = data[y0, x1]
            c = data[p1, x0]
            d = data[p1, x1]
            e = data[y1, x0]
            f = data[y1, x1]
            return (d-c-b+a) - (f-e-d+c)
        elif feat_type == Feature.HORR_3:
            dx = (x1 - x0) / 3
            p1 = int(x0 + dx)
            p2 = int(x1 - dx)
            a = data[y0, x0]
            b = data[y1, x0]
            c = data[y0, p1]
            d = data[y1, p1]
            e = data[y0, p2]
            f = data[y1, p2]
            g = data[y0, x1]
            h = data[y1, x1]
            return (d-c-b+a) - (f-e-d+c) + (h-g-f+e)
        elif feat_type == Feature.VERT_3:
            dy = (y1 - y0) / 3
            p1 = int(y0 + dy)
            p2 = int(y1 - dy)
            a = data[y0, x0]
            b = data[y0, x1]
            c = data[p1, x0]
            d = data[p1, x1]
            e = data[p2, x0]
            f = data[p2, x1]
            g = data[y1, x0]
            h = data[y1, x1]
            return (d-c-b+a) - (f-e-d+c) + (h-g-f+e)
        elif feat_type == Feature.CHECKER:
            px = int((x0 + x1)/2)
            py = int((y0 + y1)/2)
            a = data[y0, x0]
            b = data[py, x0]
            c = data[y1, x0]
            d = data[y0, px]
            e = data[py, px]
            f = data[y1, px]
            g = data[y0, x1]
            h = data[py, x1]
            i = data[y1, x1]
            return ((e-d-b+a)+(i-h-f+e)) - ((f-e-c+b)+(h-g-e+d))   #10% faster than simplifying with multiplications
        else:
            raise RuntimeError("Invalid Feature")

    @staticmethod
    def calculateNorm(sample, x0, y0, x1, y1, feat_type, mean, std):  # About 35% to 45% faster than an iterative method
        data = sample.integral
        if feat_type == Feature.HORR_2:
            p1 = int((x0 + x1) / 2)
            a = data[y0, x0]
            b = data[y1, x0]
            c = data[y0, p1]
            d = data[y1, p1]
            e = data[y0, x1]
            f = data[y1, x1]
            return ((d-c-b+a)-(f-e-d+c))/(2*std) * 128
        elif feat_type == Feature.VERT_2:
            p1 = int((y0 + y1) / 2)
            a = data[y0, x0]
            b = data[y0, x1]
            c = data[p1, x0]
            d = data[p1, x1]
            e = data[y1, x0]
            f = data[y1, x1]
            return ((d-c-b+a)-(f-e-d+c))/(2*std) * 128
        elif feat_type == Feature.HORR_3:
            dx = (x1 - x0) / 3
            p1 = int(x0 + dx)
            p2 = int(x1 - dx)
            a = data[y0, x0]
            b = data[y1, x0]
            c = data[y0, p1]
            d = data[y1, p1]
            e = data[y0, p2]
            f = data[y1, p2]
            g = data[y0, x1]
            h = data[y1, x1]
            area = (dx)*(y1-y0)
            return (((d-c-b+a)-(f-e-d+c)+(h-g-f+e) - area*mean)/(2*std) + area) * 128
        elif feat_type == Feature.VERT_3:
            dy = (y1 - y0) / 3
            p1 = int(y0 + dy)
            p2 = int(y1 - dy)
            a = data[y0, x0]
            b = data[y0, x1]
            c = data[p1, x0]
            d = data[p1, x1]
            e = data[p2, x0]
            f = data[p2, x1]
            g = data[y1, x0]
            h = data[y1, x1]
            area = (x1-x0)*(dy)
            return (((d-c-b+a)-(f-e-d+c)+(h-g-f+e) - area*mean)/(2*std) + area) * 128
        elif feat_type == Feature.CHECKER:
            px = int((x0 + x1) / 2)
            py = int((y0 + y1) / 2)
            a = data[y0, x0]
            b = data[py, x0]
            c = data[y1, x0]
            d = data[y0, px]
            e = data[py, px]
            f = data[y1, px]
            g = data[y0, x1]
            h = data[py, x1]
            i = data[y1, x1]
            return ((e-d-b+a)+(i-h-f+e)-(f-e-c+b)-(h-g-e+d))/(2*std) * 128  # 10% faster than simplifying with multiplications
        else:
            raise RuntimeError("Invalid Feature")


# ========================================================================= #
# Bounds                                                                    #
# ========================================================================= #


class GlobalBounds:

    SIZE = 24
    MIN_AREA = 32
    AVAILABLE_BOUNDS = []

    @staticmethod
    def generateBounds(type, size=SIZE, min_area=MIN_AREA):
        (cells_y, cells_x) = Feature.CELLS[type]
        bounds = []
        for width in range(cells_x, size + 1, cells_x):
            for x in range(size + 1 - width):
                for height in range(cells_y, size + 1, cells_y):
                    for y in range(size + 1 - height):
                        area = width * height
                        if area >= min_area:
                            bound = (x, y, x + width, y + height)
                            bounds.append(bound)
        return bounds

    @staticmethod
    def randomBound(feature, size=SIZE, min_size=MIN_AREA):
        cells_y, cells_x = Feature.CELLS[feature.type]
        max_x = int(size / cells_x)
        max_y = int(size / cells_y)
        width = GLOBAL_RAN.randint(1, max_x) * cells_x
        height = GLOBAL_RAN.randint(1, max_y) * cells_y
        x = GLOBAL_RAN.randint(0, size - width)
        y = GLOBAL_RAN.randint(0, size - height)
        return (x, y, x + width, y + height)

    @staticmethod
    def randomSubset(size):
        subset = []
        indices = []

        for i in range(size):
            index = GLOBAL_RAN.randint(0, len(GlobalBounds.AVAILABLE_BOUNDS)-1)
            if index in indices:
                continue
            subset.append(GlobalBounds.AVAILABLE_BOUNDS[index])
            indices.append(index)

        return (subset, indices)

    @staticmethod
    def removeIndice(index):
        assert 0 <= index and index < len(GlobalBounds.AVAILABLE_BOUNDS)
        del GlobalBounds.AVAILABLE_BOUNDS[index]

GlobalBounds.AVAILABLE_BOUNDS = [(type, bound) for type in Feature.TYPES for bound in GlobalBounds.generateBounds(type, size=GlobalBounds.SIZE, min_area=GlobalBounds.MIN_AREA)]
print("Generated: {} Global Bounds for: {} Features!".format(len(GlobalBounds.AVAILABLE_BOUNDS), len(Feature.TYPES)))


# ========================================================================= #
# Classifier                                                                #
# ========================================================================= #


class Classifier:

    def classify(self, sample, x=0, y=0, size=24):
        raise NotImplementedError('Implement Me!')

    def classifyAll(self, samples, x=0, y=0):
        classification = []
        for sample in samples:
            val = self.classify(sample, x=x, y=y)
            classification.append(val)
        return classification

    def evaluate(self, pos_samples, neg_samples, x=0, y=0):
        num_obj = len(pos_samples)
        num_neg = len(neg_samples)

        pos_classification = self.classifyAll(pos_samples, x=x, y=y)
        neg_classification = self.classifyAll(neg_samples, x=x, y=y)

        detected = sum(pos_classification)
        false_pos = sum(neg_classification)

        return (detected/num_obj, false_pos/num_neg) #, pos_classification, neg_classification)

    def test(self, pos_samples, neg_samples):
        (detected, false_pos) = (round(val*100, 2) for val in self.evaluate(pos_samples, neg_samples))
        print("\tDETECTED: {:6.2f} FALSE_POSITIVES: {:6.2f} \t[{}]".format(detected, false_pos, self))

    def detect(self, image, scale_interval=0.8, step=0.05):
        assert scale_interval < 1 and scale_interval > 0

        sample = SampleImg(image, train_sample=False)

        (height, width) = sample.integral.shape
        size = min(width, height)
        detected = []

        if size < GlobalBounds.SIZE:
            return detected

        break_flag = False
        while True:
            delta = max(size * step, 1)
            steps_x = int((width - size) / delta)
            steps_y = int((height - size) / delta)
            for i in range(steps_y):
                y = int(i * delta)
                for j in range(steps_x):
                    x = int(j * delta)
                    val = self.classify(sample, x, y, GlobalBounds.SIZE)
                    if val:
                        rect = (x, y, x+int(size), y+GlobalBounds.SIZE)
                        detected.append(rect)
            size *= scale_interval
            if size <= GlobalBounds.SIZE:
                if break_flag:
                    return detected
                size = GlobalBounds.SIZE
                break_flag = True

    def detectAverage(self, image, scale_interval=0.8, step=0.05, dist_thresh=1.5):

        detected = self.detect(image, scale_interval=scale_interval, step=step)

        unique = []
        count = []
        for j, (dx0, dy0, dx1, dy1) in enumerate(detected):
            match = False
            dxc = (dx1 + dx0) / 2
            dyc = (dy1 + dy0) / 2
            size = (dx1 - dx0 + 1)

            for i, (ux0, uy0, ux1, uy1) in enumerate(unique):
                uxc = (ux1 + ux0) / 2
                uyc = (uy1 + uy0) / 2
                dx = dxc - uxc
                dy = dyc - uyc
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < size * dist_thresh:
                    ux0 = (ux0*count[i] + dx0) / (count[i]+1)
                    uy0 = (uy0*count[i] + dy0) / (count[i]+1)
                    ux1 = (ux1*count[i] + dx1) / (count[i]+1)
                    uy1 = (uy1*count[i] + dy1) / (count[i]+1)
                    count[i] += 1
                    unique[i] = (ux0, uy0, ux1, uy1)
                    match = True
                    # break # According to algorithm this shouldn't be here. Hack to eliminate overlaps

            if not match:
                append = (dx0, dy0, dx1, dy1)
                unique.append(append)
                count.append(0)

        return [(int(b) for b in bound) for bound in unique]


# ========================================================================= #
# Classifier - WEAK                                                         #
# ========================================================================= #


class ClassifierWeak(Classifier):

    def __init__(self, feat_type, rel_bound, thresh=0.0, parity=1):
        self.feat_type = feat_type
        (self.rel_x0, self.rel_y0, self.rel_x1, self.rel_y1) = rel_bound
        self.threshold = thresh
        self.parity = parity

    def __str__(self):
        return "(Weak) {} {} {:13.10f} {}".format(((self.rel_x0, self.rel_y0, self.rel_x1, self.rel_y1)), self.feat_type, self.threshold, self.parity)

    def value(self, sample, x=0, y=0, size=GlobalBounds.SIZE):

        if sample.train_sample:
            val = Feature.calculate(sample, self.rel_x0, self.rel_y0, self.rel_x1, self.rel_y1, self.feat_type)
            # return  val / ((self.rel_x1-self.rel_x0)*(self.rel_y1-self.rel_y0))
            return val
        else:
            scale = size / GlobalBounds.SIZE
            x0 = int(x + self.rel_x0 * scale)
            y0 = int(y + self.rel_y0 * scale)
            x1 = int(x + self.rel_x1 * scale)
            y1 = int(y + self.rel_y1 * scale)
            (mean, std) = sample.meanAndStd(x, y, x+size-1, y+size-1)

            val = Feature.calculateNorm(sample, x0, y0, x1, y1, self.feat_type, mean, std)
            # return val / ((x1 - x0) * (y1 - y0))
            return val

            # sum_norm = ((sum - area*mean)/(2*std) + area) * 128
            # feat: 0
            # p1 = int((x0 + x1)/2)
            # val_norm = sample.sumNorm(x0, y0, p1, y1, mean, std) - sample.sumNorm(p1, y0, x1, y1, mean, std)

    def classify(self, sample, x=0, y=0, size=GlobalBounds.SIZE):
        val = self.value(sample, x=x, y=y, size=size)
        condition = (self.parity * val < self.parity * self.threshold)
        return 1 if condition else 0

    def train(self, pos_samples, neg_samples, pos_weightings, neg_weightings):
        num_pos = len(pos_samples)
        num_neg = len(neg_samples)
        num = num_pos + num_neg

        pos_values = np.empty_like(pos_weightings, dtype=np.float)
        neg_values = np.empty_like(neg_weightings, dtype=np.float)

        # calculte values, and thresholds and parity. keep values for reclassification
        for i in range(num_pos):
            pos_values[i] = self.value(pos_samples[i])
        for i in range(num_neg):
            neg_values[i] = self.value(neg_samples[i])

        sum_pos = np.sum(pos_values)  # faster than anything in above loops...
        sum_neg = np.sum(neg_values)

        mean_pos = sum_pos / num_pos
        mean = float( (sum_pos + sum_neg) / num )

        self.parity = (1 if mean_pos < mean else -1)
        self.threshold = mean

        # reclassify
        pos_classified = np.empty_like(pos_weightings, dtype=np.int)
        neg_classified = np.empty_like(neg_weightings, dtype=np.int)
        thresh = self.parity * self.threshold

        for i, val in enumerate(pos_values):
            pos_classified[i] = 1 if (self.parity * val < thresh) else 0
        for i, val in enumerate(neg_values):
            neg_classified[i] = 1 if (self.parity * val < thresh) else 0

        pos_errors = pos_weightings * (1 - pos_classified)
        neg_errors = neg_weightings * neg_classified
        error = np.sum(pos_errors) + np.sum(neg_errors)

        return (error, pos_classified, neg_classified)

    @staticmethod
    def computeBestFeatureBrute(pos_samples, neg_samples, pos_weights, neg_weights):
        min_error = float("inf")
        min_bound_index = None
        min_classifier = None
        min_pos_classified = None
        min_neg_classified = None

        bounds = GlobalBounds.AVAILABLE_BOUNDS
        progress = 0

        for i, (feat_type, bound) in enumerate(tqdm(bounds, desc='Evaluating All Features')):

            val = int(100 * (i + 1) / len(bounds))
            if (progress < val):
                progress = val

            classifier = ClassifierWeak(feat_type, bound)
            (error, pos_classified, neg_classified) = classifier.train(pos_samples, neg_samples, pos_weights, neg_weights)
            # update minimum
            if (error < min_error):
                min_error = error
                min_bound_index = i
                min_classifier = classifier
                min_pos_classified = np.copy(pos_classified)
                min_neg_classified = np.copy(neg_classified)

        GlobalBounds.removeIndice(min_bound_index)
        return (min_error, min_classifier, min_pos_classified, min_neg_classified)

    @staticmethod
    def computeBestFeatureBruteSubset(pos_samples, neg_samples, pos_weights, neg_weights, subset_size_ratios=(0.1, 0.1), num_best_features=1000):

        pos_size = int(len(pos_samples)*subset_size_ratios[0]) + 1
        neg_size = int(len(neg_samples)*subset_size_ratios[1]) + 1

        arg_pos = np.argsort(-pos_weights)[0:pos_size]
        arg_neg = np.argsort(-neg_weights)[0:neg_size]

        pos_samples_reduced = [pos_samples[i] for i in arg_pos]
        neg_samples_reduced = [neg_samples[i] for i in arg_neg]
        pos_weights_reduced = [pos_weights[i] for i in arg_pos]
        neg_weights_reduced = [neg_weights[i] for i in arg_neg]

        print("\t\t{}s :\tBrute Forcing {} Features on {}p + {}n Image Subsets: ".format(round(GLOBAL_TIMER.time()), len(GlobalBounds.AVAILABLE_BOUNDS), len(arg_pos), len(arg_neg)), end="", flush=True)

        reduced_features = ClassifierWeak.computeFeatures(
            pos_samples_reduced,
            neg_samples_reduced,
            pos_weights_reduced,
            neg_weights_reduced,
            range(len(GlobalBounds.AVAILABLE_BOUNDS)-1)
        )

        print("\t\t{}s :\tEvaluating {} Best Features on all {}p + {}n Images: ".format(round(GLOBAL_TIMER.time()), num_best_features, len(pos_samples), len(neg_samples)), end="", flush=True)

        reduced_features = sorted(reduced_features, key=lambda x: x[1])[0:num_best_features]

        features = ClassifierWeak.computeFeatures(
            pos_samples,
            neg_samples,
            pos_weights,
            neg_weights,
            [index for (index, _, _, _, _) in reduced_features]
        )

        (index, error, classifier, pos_classified, neg_classified) = sorted(features, key=lambda x: x[1])[0]
        GlobalBounds.removeIndice(index)
        return (error, classifier, pos_classified, neg_classified)

    @staticmethod
    def computeBestFeatureRandom(pos_samples, neg_samples, pos_weights, neg_weights, num_ran=50):
        min_error = float("inf")
        min_bound_index = None
        min_classifier = None
        min_pos_classified = None
        min_neg_classified = None

        bounds, indices = GlobalBounds.randomSubset(num_ran)

        print("\t\t{}s :\tChecking {} Random Features on {}p + {}n Image Subsets: ".format(round(GLOBAL_TIMER.time()), num_ran, len(pos_samples), len(neg_samples)), end="", flush=True)

        for i, ((feat_type, bound), index) in enumerate(zip(bounds, tqdm(indices))):

            classifier = ClassifierWeak(feat_type, bound)
            (error, pos_classified, neg_classified) = classifier.train(pos_samples, neg_samples, pos_weights, neg_weights)

            # update minimum
            if (error < min_error):
                min_error = error
                min_bound_index = index
                min_classifier = classifier
                min_pos_classified = np.copy(pos_classified)
                min_neg_classified = np.copy(neg_classified)

        GlobalBounds.removeIndice(min_bound_index)

        return (min_error, min_classifier, min_pos_classified, min_neg_classified)


    @staticmethod
    def computeFeatures(pos_samples, neg_samples, pos_weights, neg_weights, global_feature_indices):
        feature_list = []

        progress = 0

        for i, global_index in enumerate(tqdm(global_feature_indices, desc='Computing Features')):
            classifier = ClassifierWeak(*GlobalBounds.AVAILABLE_BOUNDS[global_index])
            (error, pos_classified, neg_classified) = classifier.train(pos_samples, neg_samples, pos_weights, neg_weights)
            append = (global_index, error, classifier, pos_classified, neg_classified)
            feature_list.append(append)

        return feature_list


# ========================================================================= #
# Classifier - STRONG                                                       #
# ========================================================================= #


class ClassifierStrong(Classifier):

    def __init__(self):
        self.alpha_list = []
        self.classifier_list = []
        self.threshold = 0
        self.sum_alpha = 0

    def __str__(self):
        return "(Strong) #{} alpha sum {} A{}".format(len(self.classifier_list), sum(self.alpha_list), [str(c) for c in self.classifier_list])

    def __len__(self):
        return len(self.classifier_list)

    def classify(self, sample, x=0, y=0, size=GlobalBounds.SIZE):
        sum_weak = 0
        for h, a in zip(self.classifier_list, self.alpha_list):
            sum_weak += a * h.classify(sample, x=x, y=y, size=size)
            if sum_weak >= self.threshold:
                return 1
        return 0

    def train(self, pos_samples, neg_samples, num_features, ran_tries=10):
        self._trainInit(pos_samples, neg_samples)
        for t in range(num_features):
            (_, _, error) = self._trainStep(num_ran=ran_tries)
            print('\tStrong: Train Step:', t, "Error:", error)
        self._trainEnd()
        return self

    def _trainInit(self, pos_samples, neg_samples):
        self._num_pos = len(pos_samples)
        self._num_neg = len(neg_samples)
        self._num = self._num_pos + self._num_neg
        self._pos_samples = pos_samples
        self._neg_samples = neg_samples
        self._pos_weightings = np.array( self._num_pos*[1/(2*self._num_pos)] )
        self._neg_weightings = np.array( self._num_neg*[1/(2*self._num_neg)] )
        self._pos_weightings = np.array( [ (1+GLOBAL_RAN.random()/10000)/(2*self._num_pos) for i in range(self._num_pos) ] )
        self._neg_weightings = np.array( [ (1+GLOBAL_RAN.random()/10000)/(2*self._num_neg) for i in range(self._num_neg) ] )
        return self

    def _trainStep(self, num_ran=10, subset_size_ratios=(0.1, 0.1), num_best_features=400):
        # [1. Normalise]
        sum = np.sum(self._pos_weightings) + np.sum(self._neg_weightings)
        self._pos_weightings /= sum
        self._neg_weightings /= sum

        # [2. Train Classifiers]
        if (num_ran is None) and (subset_size_ratios is not None) and (num_best_features is not None):
            (error, classifier, pos_classified, neg_classified) = ClassifierWeak.computeBestFeatureBruteSubset(self._pos_samples, self._neg_samples, self._pos_weightings, self._neg_weightings, subset_size_ratios=subset_size_ratios, num_best_features=num_best_features)
        elif (num_ran is not None) and (subset_size_ratios is None) and (num_best_features is None):
            (error, classifier, pos_classified, neg_classified) = ClassifierWeak.computeBestFeatureRandom(self._pos_samples, self._neg_samples, self._pos_weightings, self._neg_weightings, num_ran=num_ran)
        else:
            (error, classifier, pos_classified, neg_classified) = ClassifierWeak.computeBestFeatureBrute(self._pos_samples, self._neg_samples, self._pos_weightings, self._neg_weightings)

        # [4. Update Weightings]
        b = error / (1 - error)
        alpha = log(1 / b)

        for i in range(self._num_pos):
            e = 1 if pos_classified[i] == 1 else 0
            self._pos_weightings[i] *= b ** e

        for i in range(self._num_neg):
            e = 1 if neg_classified[i] == 0 else 0
            self._neg_weightings[i] *= b ** e

        # [Append Classifier]
        self.classifier_list.append(classifier)
        self.alpha_list.append(alpha)
        self.sum_alpha = float(np.sum(self.alpha_list))

        return (classifier, alpha, error)

    def _trainEnd(self):
        del self._num_pos
        del self._num_neg
        del self._num
        del self._pos_samples
        del self._neg_samples
        del self._pos_weightings
        del self._neg_weightings
        return self

    def top(self):
        return self.classifier_list[-1]


# ========================================================================= #
# Classifier - CASCADE                                                      #
# ========================================================================= #


class ClassifierCascade(Classifier):

    def __init__(self):
        self.layers = []

    def __str__(self):
        return "(Cascade) L: {}".format(["{}: {}F".format(i, len(layer)) for i, layer in enumerate(self.layers)])

    def classify(self, sample, x=0, y=0, size=24):
        for i, classifier in enumerate(self.layers):
            val = classifier.classify(sample, x=x, y=y, size=size)
            if val != 1:
                return 0
        return 1

    def train(self, pos_all_samples, neg_all_samples, min_dr=0.9, max_fp=0.5, target_fp=0.05, search_thresh=0.0001, num_ran=10, subset_size_ratios=(0.1, 0.1), num_best_features=1000):
        pos_train_samples = pos_all_samples
        neg_train_samples = neg_all_samples

        print()
        print("Training Cascade!")
        print("\t- Total Positive: {}".format(len(pos_all_samples)))
        print("\t- Total Negative: {}".format(len(neg_all_samples)))
        print("\t- Detection Rate (Layer Min): {}".format(min_dr))
        print("\t- False Positive Rate (Layer Max): {}".format(max_fp))
        print("\t- False Positive Rate (Cascade Target): {}".format(target_fp))
        print()

        fP = 1
        dR = 1
        layer_index = 0
        total_features = 0

        #Train Cascade
        while (fP > target_fp) and (len(neg_train_samples) > 0):
            layer_index += 1
            num_features = 0

            newFP = fP
            newDR = dR

            print("Training Layer: {}".format(layer_index))
            layer = ClassifierStrong()._trainInit(pos_train_samples, neg_train_samples)
            self.layers.append(layer)

            #Train Layer
            # while newFP > (max_fp * fP):
            while newFP > (max_fp * fP) or newDR < (min_dr * dR):

                #train step
                num_features += 1
                total_features += 1
                layer._trainStep(num_ran=num_ran, subset_size_ratios=subset_size_ratios, num_best_features=num_best_features)

                # binary search for threshold
                lower = 0
                upper = layer.sum_alpha
                layer.threshold = (lower + upper) / 2
                while abs(lower - upper) > search_thresh:
                    (newDR, newFP) = self.evaluate(pos_all_samples, neg_all_samples)
                    lower = layer.threshold if newDR >= min_dr*dR else lower
                    upper = layer.threshold if newDR <= min_dr*dR else upper
                    layer.threshold = (lower + upper) / 2
                (newDR, newFP) = self.evaluate(pos_all_samples, neg_all_samples)

                print("\t{:3d} -> {:3d} Detected: {:10.8f} False Positives: {:10.8f}".format(layer_index, num_features, newDR, newFP))

            #End layer Training
            layer._trainEnd()
            fP = newFP
            dR = newDR

            # Remove correctly classified training samples
            if newFP > target_fp:
                neg_train_samples = [integral for integral in neg_train_samples if layer.classify(integral) != 0]


# ========================================================================= #
# Classifier - STRONNG                                                      #
# ========================================================================= #


class ClassifierCascade(Classifier):

    def __init__(self):
        self.layers = []

    def __str__(self):
        return "(Cascade) L: {}".format(["{}: {}F".format(i, len(layer)) for i, layer in enumerate(self.layers)])

    def classify(self, sample, x=0, y=0, size=24):
        for i, classifier in enumerate(self.layers):
            val = classifier.classify(sample, x=x, y=y, size=size)
            if val != 1:
                return 0
        return 1

    def train(self, pos_all_samples, neg_all_samples, min_dr=0.9, max_fp=0.5, target_fp=0.05, search_thresh=0.0001, num_ran=10, subset_size_ratios=(0.1, 0.1), num_best_features=1000):
        pos_train_samples = pos_all_samples
        neg_train_samples = neg_all_samples

        print()
        print("Training Cascade!")
        print("\t- Total Positive: {}".format(len(pos_all_samples)))
        print("\t- Total Negative: {}".format(len(neg_all_samples)))
        print("\t- Detection Rate (Layer Min): {}".format(min_dr))
        print("\t- False Positive Rate (Layer Max): {}".format(max_fp))
        print("\t- False Positive Rate (Cascade Target): {}".format(target_fp))
        print()

        fP = 1
        dR = 1
        layer_index = 0
        total_features = 0

        #Train Cascade
        while (fP > target_fp) and (len(neg_train_samples) > 0):
            layer_index += 1
            num_features = 0

            newFP = fP
            newDR = dR

            print("Training Layer: {}".format(layer_index))
            layer = ClassifierStrong()._trainInit(pos_train_samples, neg_train_samples)
            self.layers.append(layer)

            #Train Layer
            # while newFP > (max_fp * fP):
            while newFP > (max_fp * fP) or newDR < (min_dr * dR):

                #train step
                num_features += 1
                total_features += 1
                layer._trainStep(num_ran=num_ran, subset_size_ratios=subset_size_ratios, num_best_features=num_best_features)

                # binary search for threshold
                lower = 0
                upper = layer.sum_alpha
                layer.threshold = (lower + upper) / 2
                while abs(lower - upper) > search_thresh:
                    (newDR, newFP) = self.evaluate(pos_all_samples, neg_all_samples)
                    lower = layer.threshold if newDR >= min_dr*dR else lower
                    upper = layer.threshold if newDR <= min_dr*dR else upper
                    layer.threshold = (lower + upper) / 2
                (newDR, newFP) = self.evaluate(pos_all_samples, neg_all_samples)

                print("\t{:3d} -> {:3d} Detected: {:10.8f} False Positives: {:10.8f}".format(layer_index, num_features, newDR, newFP))

            #End layer Training
            layer._trainEnd()
            fP = newFP
            dR = newDR

            # Remove correctly classified training samples
            if newFP > target_fp:
                neg_train_samples = [integral for integral in neg_train_samples if layer.classify(integral) != 0]

