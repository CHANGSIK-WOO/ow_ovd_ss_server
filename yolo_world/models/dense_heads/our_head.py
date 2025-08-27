# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer



class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """

    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        # x = x * self.logit_scale.exp() + self.bias
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # log(1 / 0.07) = Softmax Temperature ~ 14.28 = role is to scale the logits
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""

        # normalize the image and text features w.r.t the channel dimension
        x = F.normalize(x, dim=1, p=2) # x = image features (b, c, h, w) b = batch size, c = channels, h = height, w = width
        w = F.normalize(w, dim=-1, p=2) # w = texXt features (b, k, c) b = batch size, k = number of text features, c = channels

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w) # b(hw)c @ bck -> b(hw)k, matmul is matrix multiplication in each batch
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        # logits = (img_feat @ txt_teat.T) * exp(logit_scale)
        return x



class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x

@MODELS.register_module()
class OurHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias. cls_pred[-1] : last conv layer(class logit) of cls_pred
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList() #class predictions layer [batch, num_classes, H, W]
        self.reg_preds = nn.ModuleList() #Regression(Bounding Box Regression) Predictions layer [batch, 4, H, W]. 4 = [cx, cy, w, h] or [l, t, r, b]
        self.cls_contrasts = nn.ModuleList() #Classification Contrastive layer [batch, embed_dim : 256 ~ 1024, H, W]
        # NCHW [Number, channel, Height, Width]
        # (1) NLP : [batch_size, Seq_len : Token #, Embedding_dim]
        # (2) flatten data : [batch_size, num_anchors, num_channels]
        # (3) Channel size : [batch_size, Sequence_length, channel_size(H*W)] 

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float) # [0, 1, ... , reg_max - 1]
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

@MODELS.register_module()
class OurHead(YOLOv8Head):
    """YOLO-World Head
    """

    def __init__(self, world_size=-1, 
                    att_embeddings=None,
                    prev_intro_cls=0,
                    cur_intro_cls=0,
                    thr=0.8,
                    alpha=0.5,
                    use_sigmoid=True,
                    device='cuda',
                    prev_distribution=None,
                    distributions=None,
                    top_k=10,
                    *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thr = thr
        self.world_size = world_size
        self.device = device
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.distributions = distributions
        # self.thrs = [t/100.0 for t in range(50, 100, 5)]
        self.thrs = [thr]
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.prev_distribution = prev_distribution
        self.top_k = top_k
        self.load_att_embeddings(att_embeddings)
    
    def disable_log(self):
        self.positive_distributions = None
        self.negative_distributions = None
        print('disable log')
    
    def enable_log(self):
        self.reset_log()
        print('enable log')
    
    def load_att_embeddings(self, att_embeddings):
        if att_embeddings is None:
            self.att_embeddings = None
            self.disable_log()
            return
        atts = torch.load(att_embeddings)
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding']
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
        
    """YOLO World v8 head."""
    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool=False) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""
        outs = self(img_feats, txt_feats)
        # do not use att_embeddings
        if self.att_embeddings is None:
            loss_inputs = outs + (None, batch_data_samples['bboxes_labels'],
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
        loss_inputs = outs + (att_outs, batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions
   

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                fusion_att: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        if self.att_embeddings.shape[0] != 25 * (self.num_classes):
            self.select_att()
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        # outs = self.fomo_update_outs(outs)
        if self.att_embeddings is None:
            predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
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
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            att_scores: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        
        
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

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

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
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def log_distribution(self, att_scores, assigned_scores):
        
        if not self.training or self.positive_distributions is None \
            or self.att_embeddings is None:
            return
        
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
    
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
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
        
        if self.att_embeddings is not None:
            flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        else:
            flatten_cls_scores = torch.cat(flatten_att_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness, img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
