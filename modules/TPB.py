
import torch
import torch.nn as nn

from typing import Optional, List

class TPBNet(nn.Module):
    def __init__(self, num_cross_proj_layers: int = 2, clip_v_dim: int = 1024):
        super().__init__()

        layer_list = []
        for i in range(num_cross_proj_layers):
            layer_list +=[nn.Linear(clip_v_dim, clip_v_dim), nn.LayerNorm(clip_v_dim), nn.LeakyReLU()]
        layer_list += [nn.Linear(clip_v_dim, 768)]
        self.visual_projection = nn.Sequential(*layer_list)


    def forward(
        self,
        clip_vision_outputs: Optional[torch.FloatTensor] = None,
        use_global: Optional[bool] = False,
        layer_ids: Optional[List[int]] = [24],
        batch_index: Optional[int] = None,
    ):
        # convert layer_ids to list
        if isinstance(layer_ids, int):
            layer_ids = [layer_ids]
        if len(layer_ids) > 1:
            # TODO: support multiple layers
            pass
        else:
            layer_id = layer_ids[0]
            assert layer_id >= 0 and layer_id < 25, "layer_id must be in [0, 24]"
            if use_global:
                # projection_input = clip_vision_outputs.hidden_states[layer_id]
                projection_input = clip_vision_outputs.pooler_output.unsqueeze(1)
            else:
                if batch_index is not None:
                    projection_input = clip_vision_outputs.hidden_states[layer_id][batch_index, 1:, :].unsqueeze(0)
                else:
                    projection_input = clip_vision_outputs.hidden_states[layer_id][:, 1:, :]

        image_embeds = self.visual_projection(projection_input)

        return image_embeds

