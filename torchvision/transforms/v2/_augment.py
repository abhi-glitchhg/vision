import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Tuple

import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F

from ._transform import _RandomApplyTransform, Transform
from ._utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size


class RandomErasing(_RandomApplyTransform):
    """[BETA] Randomly select a rectangle region in the input image or video and erase its pixels.

    .. v2betastatus:: RandomErasing transform

    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
        p (float, optional): probability that the random erasing operation will be performed.
        scale (tuple of float, optional): range of proportion of erased area against input image.
        ratio (tuple of float, optional): range of aspect ratio of erased area.
        value (number or tuple of numbers): erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
        inplace (bool, optional): boolean to make this transform inplace. Default set to False.

    Returns:
        Erased input.

    Example:
        >>> from torchvision.transforms import v2 as transforms
        >>>
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    _v1_transform_cls = _transforms.RandomErasing

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        return dict(
            super()._extract_params_for_v1_transform(),
            value="random" if self.value is None else self.value,
        )

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        inplace: bool = False,
    ):
        super().__init__(p=p)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        self.scale = scale
        self.ratio = ratio
        if isinstance(value, (int, float)):
            self.value = [float(value)]
        elif isinstance(value, str):
            self.value = None
        elif isinstance(value, (list, tuple)):
            self.value = [float(v) for v in value]
        else:
            self.value = value
        self.inplace = inplace

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask)):
            warnings.warn(
                f"{type(self).__name__}() is currently passing through inputs of type "
                f"tv_tensors.{type(inpt).__name__}. This will likely change in the future."
            )
        return super()._call_kernel(functional, inpt, *args, **kwargs)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img_c, img_h, img_w = query_chw(flat_inputs)

        if self.value is not None and not (len(self.value) in (1, img_c)):
            raise ValueError(
                f"If value is a sequence, it should have either a single value or {img_c} (number of inpt channels)"
            )

        area = img_h * img_w

        log_ratio = self._log_ratio
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if self.value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(self.value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            break
        else:
            i, j, h, w, v = 0, 0, img_h, img_w, None

        return dict(i=i, j=j, h=h, w=w, v=v)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["v"] is not None:
            inpt = self._call_kernel(F.erase, inpt, **params, inplace=self.inplace)

        return inpt


class _BaseMixUpCutMix(Transform):
    def __init__(self, *, alpha: float = 1.0, num_classes: int, labels_getter="default") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

        self.num_classes = num_classes

        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
            raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        elif labels.ndim != 1:
            raise ValueError(
                f"labels tensor should be of shape (batch_size,) " f"but got shape {labels.shape} instead."
            )

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        label = one_hot(label, num_classes=self.num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


class MixUp(_BaseMixUpCutMix):
    """[BETA] Apply MixUp to the provided batch of images and labels.

    .. v2betastatus:: MixUp transform

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]

        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt


class CutMix(_BaseMixUpCutMix):
    """[BETA] Apply CutMix to the provided batch of images and labels.

    .. v2betastatus:: CutMix transform

    Paper: `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/abs/1905.04899>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``CutMix()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_size(flat_inputs)

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return dict(box=box, lam_adjusted=lam_adjusted)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=params["lam_adjusted"])
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            x1, y1, x2, y2 = params["box"]
            rolled = inpt.roll(1, 0)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt


  
'''

class Mosaic(nn.Module):
    
    Mosaic Transform
    

    def __init__(self, min_frac: float = 0.25, max_frac: float = 0.75, size_limit: int = 10) -> None:
        super().__init__()
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.size_limit = size_limit

    def forward(
        self, images: torch.Tensor, boxes: List[List[List[torch.Tensor]]], labels: List[List[List[torch.Tensor]]]
    ):
        
        images (torch.Tensor) : Image Tensor of B*4*C*H*W
        boxes (List[List[torch.Tensor]]) : Bounding boxes in xyxy format.
        labels (List[List[torch.Tensor]])  : labels corresponding to the bounding boxes.
        

        # implementation  is heavily inspired from this colab notebook : https://colab.research.google.com/drive/1YWb7a_3bHqG30SoIxU5S4lHKktkRseyY?usp=sharing#scrollTo=yHsYf3Z3bujO

        if images.ndim != 5:
            raise ValueError(f"Image tensor should be 5 dimensional, got {images.ndim}")
        if images.shape[1] != 4:
            raise ValueError(f"First dimension of image tensor should be of size 4, got {images.shape[1]}")

        B, C, H, W = images.shape[-2:]
        
        num_channels = images.shape[-3]

        xc = torch.randint(int(sx * self.min_frac), int(sx * self.max_frac), size=(1,))
        yc = torch.randint(int(sy * self.min_frac), int(sy * self.max_frac), size=(1,))

        x0a0, y0a0, x1a0, y1a0 = 0, 0, xc, yc
        x0b0, y0b0, x1b0, y1b0 = sx - xc, sy - yc, sx, sy

        x0a1, y0a1, x1a1, y1a1 = 0, yc, xc, sy
        x0b1, y0b1, x1b1, y1b1 = sx - xc, 0, sx, sy - yc

        x0a2, y0a2, x1a2, y1a2 = xc, 0, sx, yc
        x0b2, y0b2, x1b2, y1b2 = 0, sy - yc, sx - xc, sy

        x0a3, y0a3, x1a3, y1a3 = xc, yc, sx, sy
        x0b3, y0b3, x1b3, y1b3 = 0, 0, sx - xc, sy - yc

        mosaic_image = torch.zeros((B, num_channels, sy, sx), dtype=images.dtype)

        mosaic_image[..., y0a0:y1a0, x0a0:x1a0] = images[:, 0][..., y0b0:y1b0, x0b0:x1b0]
        mosaic_image[..., y0a1:y1a1, x0a1:x1a1] = images[:, 1][..., y0b1:y1b1, x0b1:x1b1]
        mosaic_image[..., y0a2:y1a2, x0a2:x1a2] = images[:, 2][..., y0b2:y1b2, x0b2:x1b2]
        mosaic_image[..., y0a3:y1a3, x0a3:x1a3] = images[:, 3][..., y0b3:y1b3, x0b3:x1b3]

        # calculating offsets for the bounding boxes;
        offset_y0 = y0a0 - y0b0
        offset_x0 = x0a0 - x0b0

        offset_y1 = y0a1 - y0b1
        offset_x1 = x0a1 - x0b1

        offset_y2 = y0a2 - y0b2
        offset_x2 = x0a2 - x0b2

        offset_y3 = y0a3 - y0b3
        offset_x3 = x0a3 - x0b3
        mosaic_boxes = []
        mosaic_labels = []

        for i in range(B):
            boxes[i][0][:, 0:4:2] += offset_x0
            boxes[i][0][:, 1:4:2] += offset_y0

            boxes[i][1][:, 0:4:2] += offset_x1
            boxes[i][1][:, 1:4:2] += offset_y1

            boxes[i][2][:, 0:4:2] += offset_x2
            boxes[i][2][:, 1:4:2] += offset_y2

            boxes[i][3][:, 0:4:2] += offset_x3
            boxes[i][3][:, 1:4:2] += offset_y3

            temp_box = torch.vstack(boxes[i])
            temp_label = torch.vstack(labels[i])

            temp_box[..., 0::2] = torch.clip(temp_box[..., 0::2], 0, sx)
            temp_box[..., 1::2] = torch.clip(temp_box[..., 1::2], 0, sy)

            w_ = temp_box[..., 2] - temp_box[..., 0]
            h_ = temp_box[..., 3] - temp_box[..., 1]

            mask_ = (w_ > self.size_limit) & (h_ > self.size_limit)

            temp_box = temp_box[mask_]
            temp_label = temp_label[mask_]

            mosaic_boxes.append(temp_box)
            mosaic_labels.append(temp_label)

        return mosaic_image, mosaic_boxes, mosaic_labels
'''
class Mosaic(Transform):
    """
    mosaic implemented in torchvision transform v2 api..
    should be used with the batched input
    """
    def __init__(self, p:float, min_ratio, max_ratio, size_limit, randomize_strategy):
        self.p = p 
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.randomize_strategy = randomize_strategy
        self.size_limit = size_limit

    def forward(self, x, label, boxes):
        """
        x:torch.tensor
        label:labels
        boxes:boxes
        """
        B,C,H,W = x.shape
        
        hc = torch.randint(int(H * self.min_frac), int(W * self.max_frac), size=(1,))
        wc = torch.randint(int(H * self.min_frac), int(W * self.max_frac), size=(1,))

        left_top_corner = x[...,0:hc, 0:wc]
        right_top_corner = x[...,0:hc,wc:W]
        left_bottom_corner = x[...,hc:H,0:wc]
        right_bottom_corner = x[...,hc:H,wc:W]


        left_top_shuffle, right_top_shuffle,left_bottom_shuffle,right_bottom_shuffle = torch.randperm(B),torch.randperm(B),torch.randperm(B),torch.randperm(B)

        left_top_corner = left_top_corner[left_top_shuffle]
        right_top_corner = right_top_corner[right_top_shuffle]
        left_bottom_corner = left_bottom_corner[left_bottom_shuffle]
        right_bottom_corner = right_bottom_corner[right_bottom_shuffle]

        new_x = torch.cat([torch.cat([left_top_corner, right_top_corner], dim=-1),torch.cat([left_bottom_corner,right_bottom_corner], dim=-1)],dim=-2)



        '''
        boxes: these will be a list of tensors
        '''

        

