import re
import copy
import mmcv
import numpy
import torch
import os.path as osp
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from transformers import BertTokenizer
from ..vgtr_utils.word_utils import Corpus
import numpy as np
from ..builder import PIPELINES
from transformers import XLMRobertaTokenizer

def clean_string(expression):
    return (
        re.sub(r"([.,'!?\"()*#:;])", "", expression.lower())
        .replace("-", " ")
        .replace("/", " ")
    )


@PIPELINES.register_module()
class LoadImageAnnotationsFromFile(object):
    """Load an image, referring expression, gt_bbox, gt_mask from file.

    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        dataset="RefCOCOUNC",
        color_type="color",
        backend=None,
        file_client_cfg=dict(backend="disk"),
        max_token=15,
        with_bbox=False,
        with_mask=False,
        use_token_type="default",  # bert, copus
    ):
        self.color_type = color_type
        self.backend = backend
        self.file_client_cfg = file_client_cfg.copy()
        self.file_client = None
        self.max_token = max_token
        assert with_bbox or with_mask
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        assert dataset in [
            "GRefCOCO",
            "RefCOCOUNC",
            "RefCOCOGoogle",
            "RefCOCOgUMD",
            "RefCOCOgGoogle",
            "RefCOCOPlusUNC",
            "ReferItGameBerkeley",
            "Flickr30k",
            "Mixed",
        ]
        self.dataset = dataset
        self.use_token_type = use_token_type
        if use_token_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case="uncased"
            )
        elif use_token_type == "copus":
            self.corpus = Corpus()
            corpus_path = osp.join(
                "/home/dmmm/demo_mirror/REC/vgtr/store/data/unc", "corpus.pth"
            )
            self.corpus = torch.load(corpus_path)
        elif use_token_type == "beit3":
            self.tokenizer = XLMRobertaTokenizer("pretrain_weights/beit3.spm")
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

    def _load_img(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_cfg)

        if "ReferItGame" in self.dataset or "Flickr30k" in self.dataset:
            filepath = osp.join(
                results["imgsfile"], "%d.jpg" % results["ann"]["image_id"]
            )
        elif "RefCOCO" in self.dataset:
            filepath = osp.join(
                results["imgsfile"],
                "COCO_train2014_%012d.jpg" % results["ann"]["image_id"],
            )
        elif "Mixed" == self.dataset:
            data_source = results["ann"]["data_source"]
            img_name = "COCO_train2014_%012d.jpg" if "coco" in data_source else "%d.jpg"
            img_name = img_name % results["ann"]["image_id"]
            filepath = osp.join(results["imgsfile"][data_source], img_name)
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.backend)

        results["filename"] = filepath
        results["img"] = img
        results["img_shape"] = img.shape  # (h, w, 3), rgb default
        results["ori_shape"] = img.shape
        return results

    def _load_expression(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)

        ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        for idx, word in enumerate(expression.split()):
            if word in results["token2idx"]:
                ref_expr_inds[idx] = results["token2idx"][word]
            else:
                ref_expr_inds[idx] = results["token2idx"]["UNK"]
            if idx + 1 == self.max_token:
                break

        results["ref_expr_inds"] = ref_expr_inds
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results

    def _load_expression_tokenize(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)

        # ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        # for idx, word in enumerate(expression.split()):
        #     if word in results['token2idx']:
        #         ref_expr_inds[idx] = results['token2idx'][word]
        #     else:
        #         ref_expr_inds[idx] = results['token2idx']['UNK']
        #     if idx + 1 == self.max_token:
        #         break
        encodding = self.tokenizer(
            expression,
            padding="max_length",
            truncation=True,
            max_length=self.max_token,
            return_special_tokens_mask=True,
        )
        word_id = encodding.data["input_ids"]
        word_mask = encodding.data["attention_mask"]

        results["ref_expr_inds"] = word_id
        results["text_attention_mask"] = word_mask
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results
    
    def _load_expression_tokenize_beit3(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)
        
        tokens = self.tokenizer.tokenize(expression)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")

        if len(tokens) > self.max_token - 2:
            tokens = tokens[:self.max_token - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_token - num_tokens)
        ref_expr_inds = tokens + [self.pad_token_id] * (self.max_token - num_tokens)
    
        results["ref_expr_inds"] = np.array(ref_expr_inds, dtype=int)
        results["text_attention_mask"] = np.array(padding_mask, dtype=int)
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results

    def _load_expression_copus(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)

        # ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        # for idx, word in enumerate(expression.split()):
        #     if word in results['token2idx']:
        #         ref_expr_inds[idx] = results['token2idx'][word]
        #     else:
        #         ref_expr_inds[idx] = results['token2idx']['UNK']
        #     if idx + 1 == self.max_token:
        #         break
        word_id = self.corpus.tokenize(
            expression,
            self.max_token,
        )
        word_mask = np.array(word_id > 0, dtype=int)

        results["ref_expr_inds"] = word_id
        results["text_attention_mask"] = word_mask
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results

    def _load_bbox(self, results):
        if self.with_bbox:
            gt_bbox = copy.deepcopy(results["ann"]["bbox"])
            gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
            gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
            gt_bbox = numpy.array(gt_bbox, dtype=numpy.float64)  # x1, y1, x2, y2
            h, w = results["ori_shape"][:2]
            gt_bbox[0::2] = numpy.clip(gt_bbox[0::2], 0, w - 1)
            gt_bbox[1::2] = numpy.clip(gt_bbox[1::2], 0, h - 1)
            results["gt_bbox"] = gt_bbox
        results["with_bbox"] = self.with_bbox
        return results
    
    def _load_bboxes(self, results):
        if self.with_bbox:
            gt_bboxes = copy.deepcopy(results["ann"]["bbox"][self.random_ind])
            gt_bbox_list = []
            for gt_bbox in gt_bboxes:
                gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
                gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
                gt_bbox = numpy.array(gt_bbox, dtype=numpy.float64)  # x1, y1, x2, y2
                h, w = results["ori_shape"][:2]
                gt_bbox[0::2] = numpy.clip(gt_bbox[0::2], 0, w - 1)
                gt_bbox[1::2] = numpy.clip(gt_bbox[1::2], 0, h - 1)
                gt_bbox_list.append(gt_bbox)
            results["gt_bbox"] = gt_bbox_list
        results["target"] = copy.deepcopy(results["ann"]["annotations"][self.random_ind])
        results["with_bbox"] = self.with_bbox
        return results

    def _load_mask(self, results):
        if self.with_mask:
            mask = results["ann"]["mask"]
            h, w = results["ori_shape"][:2]

            is_crowd = 0
            if isinstance(mask, list):  # polygon
                rles = maskUtils.frPyObjects(mask, h, w)
                if len(rles) > 1:
                    is_crowd = 1
                # sometimes there are multiple binary map (corresponding to multiple segs)
                rle = maskUtils.merge(rles)
            else:
                rle = mask
            mask = maskUtils.decode(rle)
            mask = BitmapMasks(mask[None], h, w)
            results["gt_mask"] = mask
            results["gt_mask_rle"] = rle  # {'size':, 'counts'}
            results["is_crowd"] = is_crowd

        results["with_mask"] = self.with_mask
        return results

    def __call__(self, results):
        results = self._load_img(results)
        if self.use_token_type=="bert":
            results = self._load_expression_tokenize(results)
        elif self.use_token_type=="copus":
            results = self._load_expression_copus(results)
        elif self.use_token_type=="beit3":
            results = self._load_expression_tokenize_beit3(results)
        else:
            results = self._load_expression(results)
        
        if self.dataset == "GRefCOCO":
            results = self._load_bboxes(results)
        else:
            results = self._load_bbox(results)
        results = self._load_mask(results)
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"file_client_args={self.file_client_cfg}, "
            f"max_token={self.max_token}, "
            f"with_bbox={self.with_bbox}, "
            f"with_mask={self.with_mask})"
        )
        return repr_str



@PIPELINES.register_module()
class LoadFromRawSource(object):
    """Load an image, referring expression, gt_bbox, gt_mask from file.

    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        dataset="RefCOCOUNC",
        color_type="color",
        backend=None,
        file_client_cfg=dict(backend="disk"),
        max_token=15,
        with_bbox=False,
        with_mask=False,
        use_token_type="default",  # bert, copus
    ):
        self.color_type = color_type
        self.backend = backend
        self.file_client_cfg = file_client_cfg.copy()
        self.file_client = None
        self.max_token = max_token
        assert with_bbox or with_mask
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        assert dataset in [
            "GRefCOCO",
            "RefCOCOUNC",
            "RefCOCOGoogle",
            "RefCOCOgUMD",
            "RefCOCOgGoogle",
            "RefCOCOPlusUNC",
            "ReferItGameBerkeley",
            "Flickr30k",
            "Mixed",
        ]
        self.dataset = dataset
        self.use_token_type = use_token_type
        if use_token_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case="uncased"
            )
        elif use_token_type == "copus":
            self.corpus = Corpus()
            corpus_path = osp.join(
                "/home/dmmm/demo_mirror/REC/vgtr/store/data/unc", "corpus.pth"
            )
            self.corpus = torch.load(corpus_path)
        elif use_token_type == "beit3":
            self.tokenizer = XLMRobertaTokenizer("/home/dmmm/demo_mirror/vlm/unilm/beit3/pretrain_weights/beit3.spm")
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

    def _load_img(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_cfg)
        filepath = results["filepath"]
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.backend)

        results["filename"] = filepath
        results["img"] = img
        results["img_shape"] = img.shape  # (h, w, 3), rgb default
        results["ori_shape"] = img.shape
        return results

    def _load_expression(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)

        ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        for idx, word in enumerate(expression.split()):
            if word in results["token2idx"]:
                ref_expr_inds[idx] = results["token2idx"][word]
            else:
                ref_expr_inds[idx] = results["token2idx"]["UNK"]
            if idx + 1 == self.max_token:
                break

        results["ref_expr_inds"] = ref_expr_inds
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results

    def _load_expression_tokenize(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)

        # ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        # for idx, word in enumerate(expression.split()):
        #     if word in results['token2idx']:
        #         ref_expr_inds[idx] = results['token2idx'][word]
        #     else:
        #         ref_expr_inds[idx] = results['token2idx']['UNK']
        #     if idx + 1 == self.max_token:
        #         break
        encodding = self.tokenizer(
            expression,
            padding="max_length",
            truncation=True,
            max_length=self.max_token,
            return_special_tokens_mask=True,
        )
        word_id = encodding.data["input_ids"]
        word_mask = encodding.data["attention_mask"]

        results["ref_expr_inds"] = word_id
        results["text_attention_mask"] = word_mask
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results
    
    def _load_expression_tokenize_beit3(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)
        
        tokens = self.tokenizer.tokenize(expression)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")

        if len(tokens) > self.max_token - 2:
            tokens = tokens[:self.max_token - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_token - num_tokens)
        ref_expr_inds = tokens + [self.pad_token_id] * (self.max_token - num_tokens)
    
        results["ref_expr_inds"] = np.array(ref_expr_inds, dtype=int)
        results["text_attention_mask"] = np.array(padding_mask, dtype=int)
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results

    def _load_expression_copus(self, results):
        expressions = results["ann"]["expressions"]
        # choice always the same if 'val'/'test'/'testA'/'testB'
        self.random_ind = np.random.choice(list(range(len(expressions))))
        expression = expressions[self.random_ind]
        expression = clean_string(expression)

        # ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        # for idx, word in enumerate(expression.split()):
        #     if word in results['token2idx']:
        #         ref_expr_inds[idx] = results['token2idx'][word]
        #     else:
        #         ref_expr_inds[idx] = results['token2idx']['UNK']
        #     if idx + 1 == self.max_token:
        #         break
        word_id = self.corpus.tokenize(
            expression,
            self.max_token,
        )
        word_mask = np.array(word_id > 0, dtype=int)

        results["ref_expr_inds"] = word_id
        results["text_attention_mask"] = word_mask
        results["expression"] = expression
        results["max_token"] = self.max_token
        return results

    def _load_bbox(self, results):
        if self.with_bbox:
            gt_bbox = copy.deepcopy(results["ann"]["bbox"])
            gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
            gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
            gt_bbox = numpy.array(gt_bbox, dtype=numpy.float64)  # x1, y1, x2, y2
            h, w = results["ori_shape"][:2]
            gt_bbox[0::2] = numpy.clip(gt_bbox[0::2], 0, w - 1)
            gt_bbox[1::2] = numpy.clip(gt_bbox[1::2], 0, h - 1)
            results["gt_bbox"] = gt_bbox
        results["with_bbox"] = self.with_bbox
        return results
    
    def _load_bboxes(self, results):
        if self.with_bbox:
            gt_bboxes = copy.deepcopy(results["ann"]["bbox"][self.random_ind])
            gt_bbox_list = []
            for gt_bbox in gt_bboxes:
                gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
                gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
                gt_bbox = numpy.array(gt_bbox, dtype=numpy.float64)  # x1, y1, x2, y2
                h, w = results["ori_shape"][:2]
                gt_bbox[0::2] = numpy.clip(gt_bbox[0::2], 0, w - 1)
                gt_bbox[1::2] = numpy.clip(gt_bbox[1::2], 0, h - 1)
                gt_bbox_list.append(gt_bbox)
            results["gt_bbox"] = gt_bbox_list
        results["target"] = copy.deepcopy(results["ann"]["annotations"][self.random_ind])
        results["with_bbox"] = self.with_bbox
        return results

    def _load_mask(self, results):
        if self.with_mask:
            mask = results["ann"]["mask"]
            h, w = results["ori_shape"][:2]

            is_crowd = 0
            if isinstance(mask, list):  # polygon
                rles = maskUtils.frPyObjects(mask, h, w)
                if len(rles) > 1:
                    is_crowd = 1
                # sometimes there are multiple binary map (corresponding to multiple segs)
                rle = maskUtils.merge(rles)
            else:
                rle = mask
            mask = maskUtils.decode(rle)
            mask = BitmapMasks(mask[None], h, w)
            results["gt_mask"] = mask
            results["gt_mask_rle"] = rle  # {'size':, 'counts'}
            results["is_crowd"] = is_crowd

        results["with_mask"] = self.with_mask
        return results

    def __call__(self, results):
        results = self._load_img(results)
        if self.use_token_type=="bert":
            results = self._load_expression_tokenize(results)
        elif self.use_token_type=="copus":
            results = self._load_expression_copus(results)
        elif self.use_token_type=="beit3":
            results = self._load_expression_tokenize_beit3(results)
        else:
            results = self._load_expression(results)
        
        if self.dataset == "GRefCOCO":
            results = self._load_bboxes(results)
        else:
            results = self._load_bbox(results)
        results = self._load_mask(results)
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"file_client_args={self.file_client_cfg}, "
            f"max_token={self.max_token}, "
            f"with_bbox={self.with_bbox}, "
            f"with_mask={self.with_mask})"
        )
        return repr_str