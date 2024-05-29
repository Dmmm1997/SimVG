import json
import numpy
from .utils import tokenize
from .builder import DATASETS
from .pipelines import Compose

# from pydantic import ListMinLengthError
from torch.utils.data.dataset import Dataset
from simvg.utils import get_root_logger, is_main

import pickle5
import lmdb
import os
import cv2
import numpy as np
from tqdm import tqdm


class BaseDataset_Ori(Dataset):
    def __init__(self, imgsfile, annsfile, pipeline, which_set="train", img_source=["coco"], word_emb_cfg=None):
        super(BaseDataset_Ori, self).__init__()
        assert isinstance(which_set, str) and which_set in [
            "train",
            "val",
            "testA",
            "testB",
            "test",
            "val_refcoco_unc",
            "testA_refcoco_unc",
            "testB_refcoco_unc",
            "val_refcocoplus_unc",
            "testA_refcocoplus_unc",
            "testB_refcocoplus_unc",
            "val_refcocog_umd",
            "test_refcocog_umd",
            "val_flickr30k",
            "val_referitgame_berkeley",
            "val_refcocog_google",
        ]
        self.which_set = which_set
        if len(img_source) == 1:
            assert img_source[0] in ["coco", "visual-genome", "flickr", "saiaprtc12"]
            self.imgsfile = imgsfile
        elif len(img_source) > 1:
            assert len(imgsfile) == len(img_source)
            assert isinstance(imgsfile, dict)
            self.imgsfile = imgsfile
        else:
            raise TypeError("None")

        self.anns_all = json.load(open(annsfile, "r"))

        self.token2idx, self.idx2token, self.word_emb = tokenize(annsfile, self.anns_all, word_emb_cfg)

        if self.anns_all["train"][0].get("data_source", None) is not None:
            self.anns_all["train"] = [ann for ann in self.anns_all["train"] if ann["data_source"] in img_source]

        if which_set == "train":
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

        if self.pipeline.transforms[0].use_token_type == "copus":
            self.num_token = len(self.pipeline.transforms[0].copus)
        elif self.pipeline.transforms[0].use_token_type == "bert":
            self.num_token = self.pipeline.transforms[0].tokenizer.vocab_size
        else:
            self.num_token = len(self.token2idx)

    def _set_group_flag(self):
        self.flag = numpy.zeros(len(self), dtype=numpy.uint8)
        for i in range(len(self)):
            ann = self.anns_all[self.which_set][i]
            if ann["width"] / ann["height"] > 1:
                self.flag[i] = 1

    def __getitem__(self, index):
        results = {"ann": self.anns_all[self.which_set][index], "which_set": self.which_set, "token2idx": self.token2idx, "imgsfile": self.imgsfile}

        results = self.pipeline(results)

        return results

    def __len__(self):
        return len(self.anns_all[self.which_set])


class BaseDataset(Dataset):
    def __init__(self, imgsfile, annsfile, pipeline, which_set="train", img_source=["coco"], word_emb_cfg=None):
        super(BaseDataset, self).__init__()
        assert isinstance(which_set, str) and which_set in [
            "train",
            "val",
            "testA",
            "testB",
            "test",
            "val_refcoco_unc",
            "testA_refcoco_unc",
            "testB_refcoco_unc",
            "val_refcocoplus_unc",
            "testA_refcocoplus_unc",
            "testB_refcocoplus_unc",
            "val_refcocog_umd",
            "test_refcocog_umd",
            "val_flickr30k",
            "val_referitgame_berkeley",
            "val_refcocog_google",
        ]
        self.which_set = which_set
        if len(img_source) == 1:
            assert img_source[0] in ["coco", "visual-genome", "flickr", "saiaprtc12"]
            self.imgsfile = imgsfile
        elif len(img_source) > 1:
            assert len(imgsfile) == len(img_source)
            assert isinstance(imgsfile, dict)
            self.imgsfile = imgsfile
        else:
            raise TypeError("None")

        self.anns_all = json.load(open(annsfile, "r"))[which_set]

        self.token2idx, self.idx2token, self.word_emb = tokenize(annsfile, self.anns_all, word_emb_cfg)

        if which_set == "train":
            if self.anns_all[0].get("data_source", None) is not None:
                self.anns_all = [ann for ann in self.anns_all if ann["data_source"] in img_source]
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

        if self.pipeline.transforms[0].use_token_type == "copus":
            self.num_token = len(self.pipeline.transforms[0].copus)
        elif self.pipeline.transforms[0].use_token_type == "bert":
            self.num_token = self.pipeline.transforms[0].tokenizer.vocab_size
        else:
            self.num_token = len(self.token2idx)

    def _set_group_flag(self):
        self.flag = numpy.zeros(len(self), dtype=numpy.uint8)
        for i in range(len(self)):
            ann = self.anns_all[i]
            if ann["width"] / ann["height"] > 1:
                self.flag[i] = 1

    def __getitem__(self, index):
        results = {"ann": self.anns_all[index], "which_set": self.which_set, "token2idx": self.token2idx, "imgsfile": self.imgsfile}

        results = self.pipeline(results)

        return results

    def __len__(self):
        return len(self.anns_all)


class BaseDatasetCRIS(Dataset):
    def __init__(self, lmdb_dir, mask_dir, imgsfile, pipeline, which_set="train"):
        super(BaseDatasetCRIS, self).__init__()
        self.which_set = which_set
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.imgsfile = imgsfile

        self._init_db()

        # if which_set == "train":
        #     self._set_group_flag()
        self.pipeline = Compose(pipeline)
        self.num_token = 1
        # self.flag = numpy.array([random.randint(0, 1) for _ in range(1000)], dtype=numpy.uint8)
        if which_set == "train":
            self.flag = numpy.zeros(len(self), dtype=numpy.uint8)
        self.word_emb = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir, subdir=os.path.isdir(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            len_buf = txn.get(b"__len__")
            keys_buf = txn.get(b"__keys__")
            if len_buf is None or keys_buf is None:
                raise ValueError("LMDB does not contain '__len__' or '__keys__'")
            self.length = self.loads_pyarrow(len_buf)
            self.keys = self.loads_pyarrow(keys_buf)

    def loads_pyarrow(self, buf):
        """
        Args:
            buf: the output of `dumps`.
        """
        return pickle5.loads(buf)

    def _set_group_flag(self):
        self.flag = numpy.zeros(len(self), dtype=numpy.uint8)
        for i in tqdm(range(len(self))):
            with self.env.begin(write=False) as txn:
                byteflow = txn.get(self.keys[i])
            ref = self.loads_pyarrow(byteflow)
            ori_img = cv2.imdecode(np.frombuffer(ref["img"], np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            if width / height > 1:
                self.flag[i] = 1

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = self.loads_pyarrow(byteflow)
        results = {"ann": ref, "which_set": self.which_set, "imgsfile": self.imgsfile, "image_id": ref["img_name"]}

        results = self.pipeline(results)

        return results

    def __len__(self):
        return self.length


@DATASETS.register_module()
class GRefCOCO(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(GRefCOCO, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"GRefCOCO-{which_set} size: {len(self)}")


@DATASETS.register_module()
class RefCOCOUNC(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(RefCOCOUNC, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"RefCOCOUNC-{which_set} size: {len(self)}")


@DATASETS.register_module()
class RefCOCOGoogle(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(RefCOCOGoogle, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"RefCOCOGoogle-{which_set} size: {len(self)}")


@DATASETS.register_module()
class RefCOCOgUMD(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(RefCOCOgUMD, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"RefCOCOg-{which_set} size: {len(self)}")


@DATASETS.register_module()
class RefCOCOgGoogle(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(RefCOCOgGoogle, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"RefCOCOg-{which_set} size: {len(self)}")


@DATASETS.register_module()
class RefCOCOPlusUNC(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(RefCOCOPlusUNC, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"RefCOCOPlusUNC-{which_set} size: {len(self)}")


@DATASETS.register_module()
class ReferItGameBerkeley(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(ReferItGameBerkeley, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"ReferItGameBerkeley-{which_set} size: {len(self)}")


@DATASETS.register_module()
class Flickr30k(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(Flickr30k, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"Flick30k-{which_set} size: {len(self)}")


@DATASETS.register_module()
class Mixed(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(Mixed, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"Mixed-{which_set} size: {len(self)}")
            # logger.info(f"Mixed tokens: {len(self.token2idx)}")


@DATASETS.register_module()
class MixedSeg(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs["which_set"]
        super(MixedSeg, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f"Mixed-{which_set} size: {len(self)}")
            # logger.info(f"Mixed tokens: {len(self.token2idx)}")
