#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
from pathlib import Path

from torch.xpu import device
from torchvision.transforms.v2 import Compose
from ultralytics.data import BaseDataset, YOLODataset

from mtgdata import ScryfallBulkType, ScryfallDataset, ScryfallImageType

print("loading...")

from ultralytics import YOLO
from ultralytics.models.yolo import detect, obb

print("Hello, World!")


class DummyGetDict:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {}


class CustomDataset(obb.OBBTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dataset = ScryfallDataset(
            img_type=ScryfallImageType.small,
            bulk_type=ScryfallBulkType.default_cards,
            ds_dir=Path(__file__).parent.parent.parent / 'mtg-dataset/mtgdata/data',
            force_update=False,
            download_mode='now',
        )

        self.labels = DummyGetDict(len(self._dataset))

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        if self.augment:
            # Training transforms
            return Compose([])
        else:
            # Val transforms
            return Compose([])

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        return


class CustomTrainer(obb.OBBTrainer):

    def __init__(self, *args, **kwargs):
        print("Custom Trainer Initialized!", args, kwargs)
        super().__init__(*args, **kwargs)

    # Custom Trainer Initialized! () {'overrides': {'model': 'yolo11n-obb.yaml', 'task': 'obb', 'data': 'DOTAv1.yaml', 'epochs': 100, 'imgsz': 1024, 'device': 'mps', 'mode': 'train'}, '_callbacks': defaultdict(<class 'list'>, {'on_pretrain_routine_start': [<function on_pretrain_routine_start at 0x127ed5260>], 'on_pretrain_routine_end': [<function on_pretrain_routine_end at 0x127ed5300>], 'on_train_start': [<function on_train_start at 0x127ed53a0>], 'on_train_epoch_start': [<function on_train_epoch_start at 0x127ed5440>], 'on_train_batch_start': [<function on_train_batch_start at 0x127ed54e0>], 'optimizer_step': [<function optimizer_step at 0x127ed5580>], 'on_before_zero_grad': [<function on_before_zero_grad at 0x127ed5620>], 'on_train_batch_end': [<function on_train_batch_end at 0x127ed56c0>], 'on_train_epoch_end': [<function on_train_epoch_end at 0x127ed5760>], 'on_fit_epoch_end': [<function on_fit_epoch_end at 0x127ed5800>], 'on_model_save': [<function on_model_save at 0x127ed58a0>], 'on_train_end': [<function on_train_end at 0x127ed5940>], 'on_params_update': [<function on_params_update at 0x127ed59e0>], 'teardown': [<function teardown at 0x127ed5a80>], 'on_val_start': [<function on_val_start at 0x127ed5b20>], 'on_val_batch_start': [<function on_val_batch_start at 0x127ed5bc0>], 'on_val_batch_end': [<function on_val_batch_end at 0x127ed5c60>], 'on_val_end': [<function on_val_end at 0x127ed5d00>], 'on_predict_start': [<function on_predict_start at 0x127ed5da0>], 'on_predict_batch_start': [<function on_predict_batch_start at 0x127ed5e40>], 'on_predict_postprocess_end': [<function on_predict_postprocess_end at 0x127ed5f80>], 'on_predict_batch_end': [<function on_predict_batch_end at 0x127ed5ee0>], 'on_predict_end': [<function on_predict_end at 0x127ed6020>], 'on_export_start': [<function on_export_start at 0x127ed60c0>], 'on_export_end': [<function on_export_end at 0x127ed6160>]})}

    def build_dataset(self, *args, **kwargs):
        a = super().build_dataset(*args, **kwargs)
        print(a.labels)
        return a



if __name__ == '__main__':
    # Create a new YOLO11n-OBB model from scratch
    model = YOLO("yolo11n-obb.yaml")
    # Train the model on the DOTAv1 dataset

    model.train(epochs=100, imgsz=640, device='mps', trainer=CustomTrainer)
