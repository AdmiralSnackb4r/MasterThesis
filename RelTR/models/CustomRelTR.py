
import torch
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .reltr import MLP, PostProcess
from .CustomMatcher import build_custom_matcher
from .CustomTransformer import build_custom_transformer
from .CustomCriterion import CustomSetCriterion
from .backbone import build_backbone

class RelTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_entites, aux_loss=False, matcher=None):
        super().__init__()
        self.num_entities = num_entites
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entites, hidden_dim*2)

        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)




    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.entity_embed.weight, pos[-1])

        output_class = self.entity_class_embed(hs)
        output_coord = self.entity_bbox_embed(hs).sigmoid()

        out = {'pred_logits':output_class[-1], 'pred_boxes':output_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    

def custom_build(args):
    num_classes = 11
    num_rel_classes = 51

    device = torch.device(args.device)
    backbone = build_backbone(args)

    transformer = build_custom_transformer(args)
    matcher = build_custom_matcher(args)
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_entites=args.num_entities,
        aux_loss=args.aux_loss,
        matcher=matcher)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef

    losses = ['labels', 'boxes']
    criterion = CustomSetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessor = {'bbox': PostProcess()}
    return model, criterion, postprocessor
