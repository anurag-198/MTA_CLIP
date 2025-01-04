# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple, Dict

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS

from mmseg.registry import MODELS

from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean

from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding

from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.dense_heads.maskformer_head import MaskFormerHead

from mmdet.models.utils import multi_apply
from mmcv.cnn import build_norm_layer ### needs norm config and embedding dimension

@MODELS.register_module()
class Mask2FormerHead_det(MaskFormerHead):
    """Extends the MMDet Mask2Former head to work with Language queries.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 60,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 text_dim: int = 1024,
                 num_classes_fin: int = 150,
                 n_contexts: int=2,
                 query_loss_type: int=0,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_queries_wt: float=1.0,
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_classes_fin
        self.n_contexts = n_contexts
        #self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.query_loss_type = query_loss_type
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.text_dim = text_dim
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        self.visual_emb_proj = Conv2d(self.text_dim, feat_channels, kernel_size=1)

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)


        self.query_embed_decoder = nn.Embedding(self.num_queries + self.n_contexts * (self.num_classes+1), feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.lang_embed = nn.Linear(feat_channels, self.num_classes + 1)
        
        ####projection for the queries ####
        self.query_project = nn.Linear(self.text_dim, feat_channels) 
        self.query_reproject = nn.Linear(feat_channels, self.text_dim)
        self.mask_reproject = nn.Linear(feat_channels, self.text_dim)

        self.norm_project = build_norm_layer(dict(type='LN'), feat_channels)[1]
        self.norm_reproject = build_norm_layer(dict(type='LN'), self.text_dim)[1]

        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_queries_wt = loss_queries_wt
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        
        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)

        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_by_feat(self, all_cls_scores: Tensor, all_queries_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``. ###########labels and corresponding masks
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]


        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_queries, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_queries_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()

        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_queries'] = losses_queries[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_queries_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_queries[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_queries'] = loss_queries_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict
    
    def _loss_by_feat_single(self, cls_scores: Tensor, queries_score: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]: 
        """Loss function for outputs from a single decoder layer. ### loss ###

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,    ## get_targets is important ########
                                        batch_gt_instances, batch_img_metas)
        

        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight) 
        cls_scores = cls_scores.flatten(0,1)
        
        lq = F.one_hot(labels, num_classes=self.num_classes+1) 
        lq = lq.unsqueeze(1).repeat((1, queries_score.shape[0], 1))
        lq = lq.permute(0,2,1)
        lq = lq.flatten(1,2) 

        queries_score =queries_score.flatten(1,2)
        queries_score = queries_score.permute(1,2,0)

        q_max, pos = torch.max(queries_score, dim=2) 
        pos_onehot = F.one_hot(pos, num_classes=self.n_contexts)
        pos_onehot = pos_onehot.flatten(1,2)

        queries_score = queries_score.flatten(1,2) 
        
        logits = torch.exp(queries_score)
        
        avg_factor=class_weight[labels].sum() 

        c_w = class_weight.unsqueeze(1).repeat(1, self.n_contexts).flatten(0,1)

        ####Case 1 - +ve is all prompts, -ve is other classes all
        if self.query_loss_type == 0:
            base = logits.sum(1).unsqueeze(1)
            sfm = logits/(base + 1e-8) ### 400x453
            loss_sfm = -1 * torch.log(sfm)
            lq_wt = lq * c_w.unsqueeze(0)
            loss_sfm = loss_sfm*lq_wt
            loss_sfm = loss_sfm * label_weights.unsqueeze(1)  
            loss_sfm = loss_sfm/lq.sum(1).unsqueeze(1)
            loss_sfm = loss_sfm.sum()/avg_factor

        ####Case 2 - +ve is max, -ve is other classes all, ensemble one
        elif self.query_loss_type == 1:
            base = logits.sum(1)
            non_max = (pos_onehot + 1)%2 
            non_max = non_max * lq
            logit_rem = logits * non_max
            logit_rem = logit_rem.sum(1)
            base = base - logit_rem
            base = base.unsqueeze(1)
            sfm = logits/(base + 1e-8) 
            loss_sfm = -1 * torch.log(sfm)
            lq_wt = lq * pos_onehot  
            lq_wt_f = lq_wt * c_w.unsqueeze(0) 
            loss_sfm = loss_sfm*lq_wt_f
            loss_sfm = loss_sfm * label_weights.unsqueeze(1)  
            loss_sfm = loss_sfm/lq_wt.sum(1).unsqueeze(1) 
            loss_sfm = loss_sfm.sum()/avg_factor
        #### case 3 - +ve is max, negative all other classes and also other from same class(non max), negative from same class
        elif self.query_loss_type == 2:
            base = logits.sum(1).unsqueeze(1)
            sfm = logits/(base + 1e-8) ### 400x453
            loss_sfm = -1 * torch.log(sfm)
            lq_wt = lq * pos_onehot  #### labels activate from GT, and pos one hot will activate the max of GT
            lq_wt_f = lq_wt * c_w.unsqueeze(0) 
            loss_sfm = loss_sfm*lq_wt_f
            loss_sfm = loss_sfm * label_weights.unsqueeze(1)  
            loss_sfm = loss_sfm/lq_wt.sum(1).unsqueeze(1) 
            loss_sfm = loss_sfm.sum()/avg_factor
        
        #### case 4 - + is max, negative is all other classes, same class others as soft negatives (maybe low loss weight)
        elif self.query_loss_type == 3:            
            pass
        
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())
       
        loss_queries = self.loss_queries_wt * loss_sfm
        
        num_total_masks = reduce_mean(cls_scores[0].new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        k = 0

        mask_preds = mask_preds[mask_weights > 0]
           
        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)

        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
           
        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )

        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        
        mask_point_targets = mask_point_targets.reshape(-1)

        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)
            


        return loss_cls, loss_queries, loss_mask, loss_dice

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor, text_embedding: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        
        decoder_out = self.transformer_decoder.post_norm(decoder_out) ## the queries, norm is on embedding dimensions 256, so good --
        
        query_mask = decoder_out[:,:self.num_queries,:].clone() ## queries for masks
        query_text = decoder_out[:,self.num_queries:,:].clone()

        query_text_project = self.query_reproject(query_text) ### projected to bigger space
        query_text_project = self.norm_reproject(query_text_project)
        
        ncontexts = len(text_embedding)
        b,n,c = query_text_project.shape 
        query_text_project = query_text_project.reshape(b, ncontexts, -1, c)
        query_text_project = query_text_project.permute(1,0,2,3)

        query_mask_project = self.mask_reproject(query_mask)
        
        text = text_embedding + self.gamma * query_text_project 
           
        text = F.normalize(text, dim=3, p=2) 
        query_mask_project = F.normalize(query_mask_project, dim=2, p=2)
        query_pred = torch.einsum('bqc,zbnc->zbqn', query_mask_project, text)
        
        cls_pred = self.cls_embed(decoder_out)
        cls_pred_final = cls_pred[:,:self.num_queries,:]
        mask_embed = self.mask_embed(decoder_out)

        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        mask_pred_final = mask_pred[:,:self.num_queries,:]
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
       
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5 
        attn_mask = attn_mask.detach()

        return cls_pred_final, query_pred, mask_pred_final, attn_mask

    def forward(self, x: List[Tensor], text_embedding: Tensor, visual_embedding: Tensor,
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.
        ## text embedding is embedding after the CLIP text encoder
        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        
        batch_size = len(batch_img_metas) 
        
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        query_embed_decoder = self.query_embed_decoder.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        
        text_embedding_projected = self.query_project(text_embedding) ### project all text embeddings to smaller space

        itern = text_embedding_projected.shape[0]

        k = 0

        cb = []
        cb.append(query_feat)
        for k in range(itern) :
            cb.append(text_embedding_projected[k])
        
        comb = torch.cat(cb, dim=1) 
        comb = self.norm_project(comb) 

        query_feat = comb
       
        mask_features, multi_scale_memorys = self.pixel_decoder(x) 
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i]) ## multiple decoder layer outputs
          
            
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1) 
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        cls_pred_list = []
        query_pred_list = []
        mask_pred_list = []

        cls_pred, query_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, text_embedding, multi_scale_memorys[0].shape[-2:])  ### chose any of text embedding 0 or 1 
            
        i = 0
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level ## feat level is 3
          
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        
            layer = self.transformer_decoder.layers[i]
            
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx], 
                query_pos=query_embed_decoder, ### positional embedding for query, learnable
                key_pos=decoder_positional_encodings[level_idx],  ### positional embedding for key, cosine embedding
                cross_attn_mask=attn_mask, ### masking regions with sigmoid from mmdet, maskformer
                query_key_padding_mask=None,
                key_padding_mask=None)
    
            cls_pred, query_score, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, text_embedding, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_pred_list.append(query_score)
            
        return cls_pred_list, query_pred_list, mask_pred_list
