# Copyright (c) Lin Song. All rights reserved.
import math, copy
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData, PixelData
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptInstanceList, OptMultiConfig, InstanceList
from mmdet.models.utils import filter_scores_and_topk, multi_apply, unpack_gt_instances
from mmyolo.models.dense_heads import YOLOv8HeadModule
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.models.dense_heads.yolov5_ins_head import ProtoModule, YOLOv5InsHead

from .yolo_world_head import ContrastiveHead, BNContrastiveHead


@MODELS.register_module()
class OurSegHeadModule(YOLOv8HeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int,
                 proto_channels: int,  # 256 channels
                 mask_channels: int,   # 32 channels
                 freeze_bbox: bool = False,
                 freeze_all: bool = False,
                 use_bn_head: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.proto_channels = proto_channels
        self.mask_channels = mask_channels
        self.freeze_bbox = freeze_bbox
        self.freeze_all = freeze_all
        self.use_bn_head = use_bn_head
        super().__init__(*args, **kwargs)
        #print("[DEBUG] INITIALIZE YOLOWorldSemSegHeadModule")

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        #print("[DEBUG] INITIALIZE WEIGHTS")
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        #print("[DEBUG] INITIALIZE LAYERS")
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.seg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        seg_out_channels = max(self.in_channels[0] // 4, self.mask_channels)
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        bbox_norm_cfg = self.norm_cfg
        bbox_norm_cfg['requires_grad'] = not self.freeze_bbox
        if self.freeze_all:
            self.norm_cfg['requires_grad'] = False
            bbox_norm_cfg['requires_grad'] = False

        for i in range(self.num_levels):
            # (B, 4 * reg_max, H, W)
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            # (B, embed_dims, H, W), embed_dims = text_channels
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            # (B, mask_channels, H, W), mask_channels = 32
            self.seg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=seg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=seg_out_channels,
                               out_channels=seg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=seg_out_channels,
                              out_channels=self.mask_channels,
                              kernel_size=1)))

            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims, self.norm_cfg))
            else:
                self.cls_contrasts.append(ContrastiveHead(self.embed_dims)) #initialization of contrastive head

        # tensor([0.0, 1.0, 2.0, ..., 15.0]). shape (16, )
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        # (B, proto_channels, H, W)
        self.proto_pred = ProtoModule(in_channels=self.in_channels[0],
                                      middle_channels=self.proto_channels,
                                      mask_channels=self.mask_channels,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg)
        if self.freeze_bbox or self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        #print("[DEBUG] FREEZE ALL")
        frozen_list = [self.cls_preds, self.reg_preds, self.cls_contrasts]
        if self.freeze_all:
            frozen_list.extend([self.proto_pred, self.seg_preds])
        for module in frozen_list:
            for m in module.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        #print("[DEBUG] TRAIN")
        super().train(mode)
        if self.freeze_bbox or self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        #print("[DEBUG] FORWARD YOLOWorldSemSegHeadModule")
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        mask_protos = self.proto_pred(img_feats[0])
        cls_logit, bbox_preds, bbox_dist_preds, coeff_preds = multi_apply(
            self.forward_single, img_feats, txt_feats, self.cls_preds,
            self.reg_preds, self.cls_contrasts, self.seg_preds)
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_preds, mask_protos
        else:
            return cls_logit, bbox_preds, None, coeff_preds, mask_protos

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       seg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        #print("[DEBUG] FORWARD SINGLE")
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)  # image feature map
        cls_logit = cls_contrast(cls_embed, txt_feat)  # cosine sim img/text
        bbox_dist_preds = reg_pred(img_feat)
        coeff_pred = seg_pred(img_feat)
        if self.reg_max > 1: # reg_max = 16
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_pred
        else:
            return cls_logit, bbox_preds, None, coeff_pred


@MODELS.register_module()
class OurSegHead(YOLOv5InsHead):
    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                             use_sigmoid=True,
                                             reduction='none',
                                             loss_weight=0.5),
                 loss_bbox: ConfigType = dict(type='IoULoss',
                                              iou_mode='ciou',
                                              bbox_format='xyxy',
                                              reduction='sum',
                                              loss_weight=7.5,
                                              return_iou=False),
                 loss_dfl=dict(type='mmdet.DistributionFocalLoss',
                               reduction='mean',
                               loss_weight=1.5 / 4),
                 mask_overlap: bool = False,
                 loss_mask: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                              use_sigmoid=True,
                                              reduction='none'),
                 loss_mask_weight=0.05,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 world_size=-1, 
                 att_embeddings=None,
                 prev_intro_cls=0,
                 cur_intro_cls=0,
                 thr=0.8,
                 alpha=0.5,
                 device="cuda",
                 prev_distribution=None,
                 distributions=None,
                 top_k=10,
                 use_sigmoid: bool = True,):
        
        super().__init__(head_module=head_module,
                         prior_generator=prior_generator,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_obj = None
        self.mask_overlap = mask_overlap
        self.loss_mask: nn.Module = MODELS.build(loss_mask)
        self.loss_mask_weight = loss_mask_weight
        self.thr = thr
        self.world_size = world_size
        self.device = device
        self.alpha = alpha
        self.distributions = distributions
        self.thrs = [thr]
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.prev_distribution = prev_distribution
        self.att_embeddings = att_embeddings # (244, 512)
        
        self.top_k = top_k
        self.positive_distributions = None
        self.negative_distributions = None
        self.use_sigmoid = use_sigmoid

        self.load_att_embeddings(att_embeddings)
        # print("[DEBUG] INITIALIZE OurSegHead")
        # print(f"self.att_embeddings = {self.att_embeddings}")
        # print(f"self.att_embeddings.shape = {self.att_embeddings.shape}") 

    def disable_log(self):
        self.positive_distributions = None
        self.negative_distributions = None
        #print('disable log')
    
    def enable_log(self):
        self.reset_log()
        #print('enable log')

    def load_att_embeddings(self, att_embeddings):
        if att_embeddings is None:
            self.att_embeddings = None
            self.disable_log()
            return
        atts = torch.load(att_embeddings)
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding']
        #print(f"self.texts : {self.texts}, self.all_atts : {self.all_atts}")

        if self.prev_distribution is not None:
            # todo this
            prev_atts_num = len(torch.load(self.prev_distribution, map_location='cuda')['positive_distributions'][self.thrs.index(self.thr)])
        else:
            prev_atts_num = 0
        self.att_embeddings = torch.nn.Parameter(atts['att_embedding'].float()[prev_atts_num:])
        # self.att_embeddings = torch.nn.Parameter(torch.zeros(1000, 512).float())
        
    def reset_log(self, interval=0.0001):
        """Reset the log."""
        # [0, 1] interval = 0.0001
        self.positive_distributions = [{att_i: torch.zeros(int((1)/interval)).to(self.device)
                                    for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions=  [{att_i: torch.zeros(int((1)/interval)).to(self.device) 
                                      for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        




    def special_init(self):
        """YOLO variants may need special initialization (assigner etc.)."""
        #print("[DEBUG] SPECIAL INIT")
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            #print(f"[DEBUG] YOLOWorldSegHead_Assigner class: {self.assigner.__class__.__name__}")
            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool=False) -> dict:
        """Forward + loss on head outputs."""
        #print("[DEBUG] LOSS")
        outs = self(img_feats, txt_feats)
        # Fast version (do not use att_embeddings)
        if self.att_embeddings is None:
            loss_inputs = outs + (None,
                                  batch_data_samples['bboxes_labels'],
                                  batch_data_samples['masks'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)
            return losses

        if fusion_att: 
            num_att = self.att_embeddings.shape[0]
            att_feats = txt_feats[:, -num_att: , :]
            txt_feats = txt_feats[:, :-num_att, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)
        
        
        with torch.no_grad():
            att_outs = self(img_feats, att_feats)[0]
        # Fast version
        loss_inputs = outs + (att_outs,
                              batch_data_samples['bboxes_labels'],
                              batch_data_samples['masks'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        coeff_preds: Optional[List[Tensor]] = None,
                        proto_preds: Optional[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Convert head features to detection & instance/semantic results."""
        #print("[DEBUG] PREDICT_BY_FEAT")
        assert len(cls_scores) == len(bbox_preds) == len(coeff_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        num_classes = cls_scores[0].size(1)
        # flatten cls_scores, bbox_preds and objectness

        if self.att_embeddings is not None:
            flatten_cls_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    num_classes)
                for cls_score in cls_scores
            ]   
        else:
            flatten_att_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    num_classes)
                for cls_score in cls_scores
            ]                             

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_coeff_preds = [
            coeff_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.head_module.mask_channels)
            for coeff_pred in coeff_preds
        ]

        if self.att_embeddings is not None:
            flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        else:
            flatten_cls_scores = torch.cat(flatten_att_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        flatten_coeff_preds = torch.cat(flatten_coeff_preds, dim=1)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            #flatten_objectness = [None for _ in range(len(featmap_sizes))] previously
            flatten_objectness = [None for _ in range(len(featmap_sizes))]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness, coeffs, mask_proto,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, flatten_coeff_preds,
                              proto_preds, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            batch_input_shape = img_meta['batch_input_shape']
            input_shape_h, input_shape_w = batch_input_shape
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']  # (top, bottom, left, right)
                input_shape_withoutpad = (input_shape_h - pad_param[0] -
                                          pad_param[1], input_shape_w -
                                          pad_param[2] - pad_param[3])
            else:
                pad_param = None
                input_shape_withoutpad = batch_input_shape
            scale_factor = (input_shape_withoutpad[1] / ori_shape[1],
                            input_shape_withoutpad[0] / ori_shape[0])

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]
                coeffs = coeffs[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]
                # NOTE: Important for mask branch
                coeffs *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                h, w = ori_shape[:2] if rescale else img_meta['img_shape'][:2]
                empty_results.masks = torch.zeros(
                    size=(0, h, w), dtype=torch.bool, device=bboxes.device)
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0], coeffs=coeffs))
                labels = results['labels']
                coeffs = results['coeffs']
            else:
                out = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(coeffs=coeffs))
                scores, labels, keep_idxs, filtered_results = out
                coeffs = filtered_results['coeffs']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                coeffs=coeffs)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            if len(results.bboxes):
                masks = self.process_mask(mask_proto, results.coeffs,
                                          results.bboxes,
                                          (input_shape_h, input_shape_w), True) # (243, 640, 640)
                print(f"masks : {masks.shape}")
                
                # Semantic inference
                soft_masks = masks.squeeze(0).float()   # (243, 640, 640)
                print(soft_masks.max(dim=0))
                if soft_masks.shape[-2:] != (input_shape_h, input_shape_w):
                    print("soft_masks.shape is not equal to input_shape")
                    soft_masks = F.interpolate(soft_masks[None], size=(input_shape_h, input_shape_w), mode='bilinear', align_corners=False).squeeze(0) # (M, Hin, Win)
                    print(f"after interpolating masks : {soft_masks.shape}")
                
                mask_num = soft_masks.shape[0] # 243
                device = soft_masks.device #cuda

                if mask_num > 0 : # mask_num = 243
                    # Even if obj*cls might already have been applied, this also safely handles the path where objectness is None

                    inst_scores = results.scores # (243,)
                    # print(f"inst_scores : {inst_scores}")
                    # print(f"inst_scores.shape : {inst_scores.shape}") 
                    
                    inst_labels = results.labels # (243,) [0, 1, 2, 3, 4, 5]
                    # print(f"inst_labels : {inst_labels}")
                    # print(f"inst_labels.shape : {inst_labels.shape}")
                    # print(f"inst_labels.np.unique : {inst_labels.unique()}")
                    
                    one_hot = F.one_hot(inst_labels.long()).to(soft_masks.dtype) # [0,1,2,3,4,5] (243, 6)
                    # print(f"one_hot.shape : {one_hot.shape}") 

                    # (M,1,1), _ : in_place operation
                    weights = inst_scores.clamp_(0, 1)[:, None, None] # (243, 1, 1)      
                    
                    weighted_masks = soft_masks * weights # (243, 640, 640)
                    # S[k,h,w] = sum_i one_hot[i,k] * weighted_masks[i,h,w]
                    S = torch.einsum('mk,mhw->khw', one_hot, weighted_masks)  # (243, 6) * (243, 640, 640) = (6, 640, 640)
                    #print(f"one_hot.shape : {one_hot.shape}, weighted_masks.shape : {weighted_masks.shape}, S.shape : {S.shape}")
                        
                    # Final class per pixel
                    best_vals, best_ids = S.max(dim=0)   # torch.max (max, max_indices) (640, 640)
                    sem_map = best_ids.to(torch.long)
                    # print(f"sem_map.unique : {sem_map.unique()}") # [0, 1, 2, 4, 5]
                    #tau = 0.02 # 0.02 --> 0.10
                    #sem_map[best_vals < tau] = 0  # set background ids = 0 
                    # print(f"after filtering sem_map.unique : {sem_map.unique()}")

                    conf_map = best_vals      

                    # If rescale, resize semantic map to original size (ori_shape) (nearest)
                    if rescale:     
                        # print("rescale")
                        if pad_param is not None:
                            top_pad, _, left_pad, _ = pad_param
                            top, left = int(top_pad), int(left_pad)
                            bottom, right = int(input_shape_h - top_pad), int(input_shape_w - left_pad)
                            sem_map = sem_map[top:bottom, left:right]  # pad crop
                            conf_map = conf_map[top:bottom, left:right]  # pad crop

                        # # print(f"[DEBUG] sem_map.shape before interpolate = {sem_map.shape}")
                        sem_map = F.interpolate(sem_map[None, None].float(),
                                                size=ori_shape[:2],
                                                mode='nearest').squeeze(0).squeeze(0).to(torch.long)
                        
                        conf_map = F.interpolate(conf_map[None, None],
                                                size=ori_shape[:2],
                                                mode='bilinear').squeeze(0).squeeze(0).to(torch.float32)                        
                        
                    # # print(f"[DEBUG] sem_map.shape after interpolate = {sem_map.shape}")
                    # # print(f"results : {results}")
                    
                    img_meta["pred_sem_seg"] = sem_map.unsqueeze(0) # Framework convention: (1, H, W)
                    img_meta["pred_sem_conf"] = conf_map.unsqueeze(0) # Framework convention: (1, H, W)
                    # # print(f"img_meta : {img_meta}")
                    # # print(f'img_meta["pred_sem_seg"].shape : {img_meta["pred_sem_seg"].shape}')
                    # # print(f'pred_sem_seg unique idx : {img_meta["pred_sem_seg"].unique().tolist()}')
                    # # print(f'img_meta["pred_sem_conf"].shape : {img_meta["pred_sem_conf"].shape}')
                    # # print(f'pred_sem_conf unique idx : {img_meta["pred_sem_conf"].unique().tolist()}')                    

                else:
                    # print("else")
                    # Empty semantic map when no instances
                    H, W = (ori_shape[0], ori_shape[1]) if rescale else (input_shape_h, input_shape_w)
                    img_meta["pred_sem_seg"] = torch.zeros((1, H, W), device=device, dtype=torch.long)                            

                if rescale:
                    if pad_param is not None:
                        # bbox minus pad param
                        top_pad, _, left_pad, _ = pad_param
                        results.bboxes -= results.bboxes.new_tensor(
                            [left_pad, top_pad, left_pad, top_pad])
                        # mask crop pad param
                        top, left = int(top_pad), int(left_pad)
                        bottom, right = int(input_shape_h -
                                            top_pad), int(input_shape_w -
                                                          left_pad)
                        masks = masks[:, :, top:bottom, left:right]
                    results.bboxes /= results.bboxes.new_tensor(
                        scale_factor).repeat((1, 2))

                    fast_test = cfg.get('fast_test', False)
                    if fast_test:
                        masks = F.interpolate(
                            masks,
                            size=ori_shape,
                            mode='bilinear',
                            align_corners=False)
                        masks = masks.squeeze(0)
                        masks = masks > cfg.mask_thr_binary
                    else:
                        masks.gt_(cfg.mask_thr_binary)
                        masks = torch.as_tensor(masks, dtype=torch.uint8)
                        masks = masks[0].permute(1, 2,
                                                 0).contiguous().cpu().numpy()
                        masks = mmcv.imresize(masks,
                                              (ori_shape[1], ori_shape[0]))

                        if len(masks.shape) == 2:
                            masks = masks[:, :, None]
                        masks = torch.from_numpy(masks).permute(2, 0, 1)

                results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
                results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

                results.masks = masks.bool()
                results_list.append(results)
            else:
                h, w = ori_shape[:2] if rescale else img_meta['img_shape'][:2]
                results.masks = torch.zeros(
                    size=(0, h, w), dtype=torch.bool, device=bboxes.device)
                results_list.append(results)
        # print(f"results_list : {results_list}")
        return results_list
    
    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Forward for loss and predictions."""
        #print("[DEBUG] LOSS_AND_PREDICT")
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances,
                              batch_img_metas,
                              batch_gt_instances_ignore)
        
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        #print("[DEBUG] FORWARD YOLOWorldSemSegHead")
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                fusion_att: bool = False) -> InstanceList:
        """Head forward + postprocess to predictions."""
        #print("[DEBUG] PREDICT")
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        if self.att_embeddings is None:
            predictions = self.predict_by_feat(*outs,
                                            batch_img_metas=batch_img_metas,
                                            rescale=rescale)
            
            for ds, meta in zip(batch_data_samples, batch_img_metas):
                # # print(f"before batch_data_samples : {batch_data_samples}")
                # # print(f"batch_img_metas : {batch_img_metas}")

                sem = meta.get("pred_sem_seg", None)
                if sem is not None :
                    if not torch.is_tensor(sem): sem = torch.as_tensor(sem)
                ds.pred_sem_seg = PixelData(data = sem.to(torch.long))

                conf = meta.get("pred_sem_conf", None)
                if conf is not None : 
                    if not torch.is_tensor(conf): conf = torch.as_tensor(conf)
                ds.pred_sem_conf = PixelData(data = conf.to(torch.float32))  

            return predictions
        
        if fusion_att: 
            num_att = self.att_embeddings.shape[0]
            att_feats = txt_feats[:, -num_att: , :]
            txt_feats = txt_feats[:, :-num_att, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)
        
        if self.att_embeddings is not None:
            outs = self.predict_unknown(outs, img_feats, att_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        
        for ds, meta in zip(batch_data_samples, batch_img_metas):
            # # print(f"before batch_data_samples : {batch_data_samples}")
            # # print(f"batch_img_metas : {batch_img_metas}")

            sem = meta.get("pred_sem_seg", None)
            if sem is not None :
                if not torch.is_tensor(sem): sem = torch.as_tensor(sem)
            ds.pred_sem_seg = PixelData(data = sem.to(torch.long))

            conf = meta.get("pred_sem_conf", None)
            if conf is not None : 
                if not torch.is_tensor(conf): conf = torch.as_tensor(conf)
            ds.pred_sem_conf = PixelData(data = conf.to(torch.float32))      
        return predictions
          

    def fomo_update_outs(self, outs):
        predictions = outs[0]
        ret_logits = []
        for prediction in predictions:
            known_logits = prediction.permute(0, 2, 3, 1)[..., :self.num_classes]
            unknown_logits = prediction.permute(0, 2, 3, 1)[..., :self.num_classes]
            unknown_logits = unknown_logits.max(-1, keepdim=True)[0]
            ret_logits.append(torch.cat([known_logits, unknown_logits], dim=-1).permute(0, 3, 1, 2))
        return (ret_logits, *outs[1:])

    def calculate_uncertainty(self, known_logits):
        known_logits = torch.clamp(known_logits, 1e-6, 1 - 1e-6)
        entropy = (-known_logits * torch.log(known_logits) - (1 - known_logits) * torch.log(1 - known_logits)).mean(dim=-1, keepdim=True)
        return entropy
    
    def select_top_k_attributes(self, adjusted_scores: Tensor, k: int = 3) -> Tensor:
        top_k_scores, _ = adjusted_scores.topk(k, dim=-1)
        top_k_average = top_k_scores.mean(dim=-1, keepdim=True)
        return top_k_average

    def compute_weighted_top_k_attributes(self, adjusted_scores: Tensor, k: int = 10) -> Tensor:
        top_k_scores, top_k_indices = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average

    def predict_unknown(self, outs, img_feats, att_embeddings):
        known_predictions = outs[0]
        unknown_predictions = self(img_feats, att_embeddings)[0]
        ret_logits = []

        for known_logits, unknown_logits in zip(known_predictions, unknown_predictions):
            known_logits = known_logits.sigmoid().permute(0, 2, 3, 1)
            unknown_logits = unknown_logits.sigmoid().permute(0, 2, 3, 1)

            # 溫←츞藥꿰윥?��?��?���쉪訝띸??�若?����
            uncertainty = self.calculate_uncertainty(known_logits)
            # uncertainty = 0
            # 溫←츞掠?���㏛?���?�若?���㎩뭉瘟껅빐掠욄�㎪?���뇥
            # top_k_att_score = self.select_top_k_attributes(unknown_logits, k=self.top_k)
            top_k_att_score = self.compute_weighted_top_k_attributes(unknown_logits, k=self.top_k)
            #top_k_att_score = unknown_logits.max(dim=-1, keepdim=True)[0]
            # �엻�릦?��꿰윥��?��?���윥?��?��?���쉪�?꾣탩
            
            unknown_logits_final = (top_k_att_score + uncertainty) / 2 * (1 - known_logits.max(-1, keepdim=True)[0])
            # unknown_logits_final = (top_k_att_score) * (1 - known_logits.max(-1, keepdim=True)[0])
            
            # �릦亮뜹?���윥��?��?���윥?��?��?���쉪����???��쥋役?��?���옖
            logits = torch.cat([known_logits, unknown_logits_final], dim=-1).permute(0, 3, 1, 2)
            ret_logits.append(logits)
        
        return (ret_logits, *outs[1:])

    def get_all_dis_sim(self, positive_dis, negative_dis):
        dis_sim = []
        for i in range(len(positive_dis)):
            positive = positive_dis[i]
            negative = negative_dis[i]
            positive = positive / positive.sum()
            negative = negative / negative.sum()
            dis_sim.append(self.get_sim(positive, negative))
        # (num_attributes,)
        return torch.stack(dis_sim).to('cuda')
        
    def combine_distributions(self):
        if self.prev_distribution is None:
            return self.positive_distributions, self.negative_distributions

        # Load previous distributions
        prev_distributions = torch.load(self.prev_distribution, map_location='cuda')
        prev_positive_distributions, prev_negative_distributions = prev_distributions['positive_distributions'], prev_distributions['negative_distributions']

        # Initialize result lists
        ret_pos, ret_neg = prev_positive_distributions, prev_negative_distributions

        # Combine distributions
        for thr in self.thrs:
            thr_id = self.thrs.index(thr)
            if thr_id >= len(prev_positive_distributions) or prev_positive_distributions[thr_id] is None:
                continue
            if thr_id >= len(self.positive_distributions) or self.positive_distributions[thr_id] is None:
                continue
            cur_pos_dist = self.positive_distributions[thr_id]
            cur_neg_dist = self.negative_distributions[thr_id]
            prev_pos_dist = prev_positive_distributions[thr_id]
            prev_neg_dist = prev_negative_distributions[thr_id]
            prev_att = len(prev_pos_dist)
            prev_pos_dist.update({prev_att + k: v for k, v in cur_pos_dist.items()})
            prev_neg_dist.update({prev_att + k: v for k, v in cur_neg_dist.items()})
            ret_pos[thr_id] = prev_pos_dist
            ret_neg[thr_id] = prev_neg_dist
        
        return ret_pos, ret_neg

    def select_att(self, per_class=25):
        """
        Select attributes based on a balance of distribution similarity and attribute diversity.
        Optimized for speed by avoiding redundant calculations and using batch operations.
        """
        
        print(f'thr: {self.thr}')
        # save_root = os.path.dirname(self.distributions)
        # task_id = self.distributions[-5]
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root)
        # torch.save({'positive_distributions': self.positive_distributions,
        #             'negative_distributions': self.negative_distributions}, os.path.join(save_root, f'current{task_id}.pth'))
        # print('save current to {}'.format(os.path.join(save_root, f'current{task_id}.pth')))
        # self.positive_distributions, self.negative_distributions = self.combine_distributions()

        # torch.save({'positive_distributions': self.positive_distributions,
        #             'negative_distributions': self.negative_distributions}, self.distributions)
        # print('save distributions to {}'.format(self.distributions))
        
        distributions = torch.load(self.distributions, map_location='cuda')
        self.positive_distributions, self.negative_distributions = distributions['positive_distributions'], distributions['negative_distributions']
        
        thr_id = self.thrs.index(self.thr)                                                            
        # Step 1: Calculate distribution similarity for each attribute (JS divergence)
        distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id], self.negative_distributions[thr_id])
        # Step 2: Prepare for batch cosine similarity calculation
        # Precompute the cosine similarities for all attribute pairs in one batch
        all_atts = self.all_atts.to(self.att_embeddings.device)
        
        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)  # Normalize embeddings
        if self.use_sigmoid:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid() 
        else:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).abs()
        
        # Initialize selected indices
        selected_indices = []
        
        # Step 3: Attribute selection loop
        for _ in range(per_class * self.num_classes):
            if len(selected_indices) == 0:
                # Select the first attribute with the lowest distribution similarity
                _, idx = distribution_sim.min(dim=0)
            else:
                # Step 4: Calculate diversity score for each unselected attribute
                # Get the mean cosine similarity between unselected and selected attributes
                unselected_indices = list(set(range(len(self.texts))) - set(selected_indices))
                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)  # Shape: (num_unselected,)
                
                # Calculate final score: balance distribution similarity and diversity (cosine similarity)
                distribution_sim_unselected = distribution_sim[unselected_indices]
                score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected
                
                # Select the attribute with the lowest score
                idx = unselected_indices[score.argmin()]
            
            selected_indices.append(idx)
        
        # Step 5: Update selected attributes and their embeddings
        selected_indices = torch.tensor(selected_indices).to(self.att_embeddings.device)
        self.att_embeddings = torch.nn.Parameter(all_atts[selected_indices]).to(self.att_embeddings.device)
        self.texts = [self.texts[i] for i in selected_indices]
                     
        print('Selected attributes saved.')
  
    def get_sim(self, a, b):
        """
            return distribution a and b similarity. lower value means more similar
        """
        def jensen_shannon_divergence(p, q):
            m = 0.5 * (p + q)
            m = m.clamp(min=1e-6)
            js_div = 0.5 * (torch.sum(p * torch.log((p / m).clamp(min=1e-6))) +
                            torch.sum(q * torch.log((q / m).clamp(min=1e-6))))
            return js_div

        return jensen_shannon_divergence(a, b)

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        #print("[DEBUG] AUG_TEST")
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            coeff_preds: Sequence[Tensor],
            proto_preds: Tensor,
            att_scores: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_gt_masks: Sequence[Tensor],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate training losses based on head features."""
        #print("[DEBUG] LOSS_BY_FEAT")
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]

        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_pred_coeffs = [
            coeff_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.head_module.mask_channels)
            for coeff_pred in coeff_preds
        ]

        if self.att_embeddings is not None:
            # att 
            flatten_att_scores = [
                att_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    att_score.shape[1])
                for att_score in att_scores
            ]
            flatten_att_scores = torch.cat(flatten_att_scores, dim=1)
        else:
            flatten_att_scores = None

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])
        flatten_pred_coeffs = torch.cat(flatten_pred_coeffs, dim=1)
        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_gt_idxs = assigned_result['assigned_gt_idxs']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        self.log_distribution(flatten_att_scores, assigned_scores)
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()

        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)

            _, c, mask_h, mask_w = proto_preds.shape
            if batch_gt_masks.shape[-2:] != (mask_h, mask_w):
                batch_gt_masks = F.interpolate(batch_gt_masks[None],
                                               (mask_h, mask_w),
                                               mode='nearest')[0]

            loss_mask = torch.zeros(1, device=loss_dfl.device)
            box_sum_flag = pad_bbox_flag.long().sum(dim=1).squeeze(1)

            batch_inds = torch.zeros(num_imgs,
                                     dtype=torch.int64,
                                     device=assigned_gt_idxs.device)[:, None]
            batch_inds[1:] = box_sum_flag.cumsum(dim=0)[:-1][..., None]
            _assigned_gt_idxs = assigned_gt_idxs + batch_inds

            for bs in range(num_imgs):
                bbox_match_inds = assigned_gt_idxs[bs]
                mask_match_inds = _assigned_gt_idxs[bs]
                bbox_match_inds = torch.masked_select(bbox_match_inds, fg_mask_pre_prior[bs])
                mask_match_inds = torch.masked_select(mask_match_inds,
                                                      fg_mask_pre_prior[bs])
                # mask
                mask_dim = coeff_preds[0].shape[1]
                prior_mask_mask = fg_mask_pre_prior[bs].unsqueeze(-1).repeat([1, mask_dim])
                pred_coeffs_pos = torch.masked_select(flatten_pred_coeffs[bs],
                                                      prior_mask_mask).reshape(
                                                          [-1, mask_dim])

                match_boxes = gt_bboxes[bs][bbox_match_inds] / 4
                normed_boxes = gt_bboxes[bs][bbox_match_inds] / 640
                bbox_area = (normed_boxes[:, 2:] -
                             normed_boxes[:, :2]).prod(dim=1)
                if not mask_match_inds.any():
                    continue
                assert not self.mask_overlap
                mask_gti = batch_gt_masks[mask_match_inds]
                mask_preds = (
                    pred_coeffs_pos @ proto_preds[bs].view(c, -1)).view(
                        -1, mask_h, mask_w)
                loss_mask_full = self.loss_mask(mask_preds, mask_gti)
                _loss_mask = (self.crop_mask(loss_mask_full[None],
                                             match_boxes).mean(dim=(2, 3)) /
                              bbox_area)
                loss_mask += _loss_mask.mean()

        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_mask = flatten_pred_coeffs.sum() * 0
        _, world_size = get_dist_info()

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size,
                    loss_mask=loss_mask * self.loss_mask_weight * world_size)

    def log_distribution(self, att_scores, assigned_scores):
        # Safely check if we should perform logging. If not, exit early.
        # print("[DEBUG] call log_distribution")
        # print(f"self.att_embeddings: {self.att_embeddings}")
        if not self.training or getattr(self, 'att_embeddings', None) is None:
            return            

        if getattr(self, 'positive_distributions', None) is None:
            self.reset_log()
        
        num_att = att_scores.shape[-1]
        num_known = assigned_scores.shape[-1]
        att_scores = att_scores.sigmoid().reshape(-1, num_att).float()      
        assigned_scores = assigned_scores.reshape(-1, num_known)
        # set previous classes to 0
        assigned_scores[:, 0: self.prev_intro_cls] = 0
        assigned_scores = assigned_scores.max(-1)[0]
        for idx, thr in enumerate(self.thrs):
            positive = (assigned_scores >= thr)
            positive_scores = att_scores[positive]
            negative_scores = att_scores[~positive]
            for att_i in range(num_att):
                self.positive_distributions[idx][att_i] += torch.histc(positive_scores[:, att_i], bins=int(1/0.0001), min=0, max=1)
                self.negative_distributions[idx][att_i] += torch.histc(negative_scores[:, att_i], bins=int(1/0.0001), min=0, max=1)