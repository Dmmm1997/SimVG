# -*- coding: utf-8 -*-

import sys
import cv2
import torch
import numpy as np
import os.path as osp
import torch.utils.data as data
from . import vgtr_utils as utils
from .vgtr_utils import Corpus
from .vgtr_utils.transforms import trans, trans_simple
from transformers import BertTokenizer
from collections import defaultdict
from torchvision.transforms import Compose, ToTensor, Normalize
from .builder import DATASETS

sys.path.append(".")
sys.modules["utils"] = utils
cv2.setNumThreads(0)

Transforms = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


@DATASETS.register_module()
class VGTRDataset(data.Dataset):

    SUPPORTED_DATASETS = {
        "grefcoco": {
            "splits": ("train", "val", "trainval", "testA", "testB"),
            "params": {"dataset": "refcoco", "split_by": "unc"},
        },
        "refcoco": {
            "splits": ("train", "val", "trainval", "testA", "testB"),
            "params": {"dataset": "refcoco", "split_by": "unc"},
        },
        "refcoco+": {
            "splits": ("train", "val", "trainval", "testA", "testB"),
            "params": {"dataset": "refcoco+", "split_by": "unc"},
        },
        "refcocog": {
            "splits": ("train", "val"),
            "params": {"dataset": "refcocog", "split_by": "google"},
        },
        "refcocog_umd": {
            "splits": ("train", "val", "test"),
            "params": {"dataset": "refcocog", "split_by": "umd"},
        },
        "flickr": {"splits": ("train", "val", "test")},
        "copsref": {"splits": ("train", "val", "test")},
    }

    # map the dataset name to data folder
    MAPPING = {
        "refcoco": "unc",
        "grefcoco": "grefcoco",
        "refcoco+": "unc+",
        "refcocog": "gref",
        "refcocog_umd": "gref_umd",
        "flickr": "flickr",
        "copsref": "copsref",
    }

    def __init__(
        self,
        data_root,
        split_root="data",
        dataset="refcoco",
        imsize=512,
        transform=None,
        testmode=False,
        split="train",
        max_query_len=20,
        augment=False,
    ):
        self.which_set = split
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.transform = Transforms  ###
        self.testmode = testmode
        self.split = split
        self.trans = trans if augment else trans_simple

        if self.dataset == "flickr":
            self.dataset_root = osp.join(self.data_root, "Flickr30k")
            self.im_dir = osp.join(self.dataset_root, "flickr30k-images")
        elif self.dataset == "copsref":
            self.dataset_root = osp.join(self.data_root, "copsref")
            self.im_dir = osp.join(self.dataset_root, "images")
        else:
            self.dataset_root = osp.join(self.data_root, "other")
            self.im_dir = osp.join(
                self.dataset_root, "images", "mscoco", "images", "train2014"
            )
            self.split_dir = osp.join(self.dataset_root, "splits")

        self.sup_set = self.dataset
        self.dataset = self.MAPPING[self.dataset]

        if not self.exists_dataset():
            print(
                "Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ"
            )
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.sup_set]["splits"]

        self.corpus = Corpus()
        corpus_path = osp.join(dataset_path, "corpus.pth")
        self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                "Dataset {0} does not have split {1}".format(self.dataset, split)
            )

        # splits = [split]
        splits = ["train", "val"] if split == "trainval" else [split]
        for split in splits:
            imgset_file = "{0}_{1}.pth".format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)
        target_guided_images = defaultdict(list)
        
        for img in self.images:
            pth = img[1]
            target_guided_images[pth].append(img)
        self.images = list(target_guided_images.values())

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case="uncased"
        )
        if self.which_set == "train":
            self._set_group_flag()
            
        self.num_token = len(self.corpus)
        self.word_emb = None

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            bbox = self.images[i][0][2]
            height = bbox[2]
            width = bbox[3]
            if height / (width+1e-6) < 1:
                self.flag[i] = 1

    def exists_dataset(self):

        return osp.exists(osp.join(self.split_root, self.dataset))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """

        :return: (img, phrase word id, phrase word mask, bounding bbox)
        """
        if self.dataset == "flickr" or self.dataset == "copsref":
            items = self.images[index]
            img_file, bbox, phrase = items[np.random.choice(len(items))]
            # original
            # img_file, bbox, phrase = self.images[index]
        else:
            items = self.images[index]
            img_file, _, bbox, phrase, _ = items[np.random.choice(len(items))]
            # original
            # img_file, _, bbox, phrase, _ = self.images[index]

        if not self.dataset == "flickr":
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)

        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)

        phrase = phrase.lower()

        img, phrase, bbox = self.trans(img, phrase, bbox, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        # tokenize phrase
        word_id = self.corpus.tokenize(phrase, self.query_len)
        word_mask = np.array(word_id > 0, dtype=int)
        # tokenize vilt
        # encodding = self.tokenizer(
        #     phrase,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.query_len,
        #     return_special_tokens_mask=True,
        # )
        # word_id = encodding.data["input_ids"]
        # word_mask = encodding.data["attention_mask"]

        return {"img": img, "ref_expr_inds": word_id, "gt_bbox": bbox}

        # return (
        #     img,
        #     np.array(word_id, dtype=int),
        #     np.array(word_mask, dtype=int),
        #     np.array(bbox, dtype=np.float32),
        # )
