import cv2
import numpy as np
def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im

class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean,
                           (list, tuple)) and isinstance(self.std,
                                                         (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)

        if label is None:
            return (im, )
        else:
            return (im, label)

class Resize:
    """
    Resize an image.

    Args:
        target_size (list|tuple, optional): The target size of image. Default: (512, 512).
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".

    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label),

        Raises:
            TypeError: When the 'img' type is not numpy.
            ValueError: When the length of "im" shape is not 3.
        """

        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = F.resize(label, self.target_size,
                                      cv2.INTER_NEAREST)

        if label is None:
            return (im, )
        else:
            return (im, label)

class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im, label=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = np.asarray(Image.open(label))
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = np.transpose(im, (2, 0, 1))
        return (im, label)

def post_process(dets, num_classes):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #print('dets',dets[0].keys())
    return dets[0]

def merge_outputs(detections, num_classes, max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def ctdet_post_process(dets, num_classes):
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

