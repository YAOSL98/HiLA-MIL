import torch
import timm
import os
# class TimmCNNEncoder(torch.nn.Module):
#     def __init__(self, model_name: str = 'resnet50.tv_in1k', 
#                  kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
#                  pool: bool = True):
#         super().__init__()
#         assert kwargs.get('pretrained', False), 'only pretrained models are supported'
#         self.model = timm.create_model(model_name, **kwargs)
#         self.model_name = model_name
#         if pool:
#             self.pool = torch.nn.AdaptiveAvgPool2d(1)
#         else:
#             self.pool = None
    
#     def forward(self, x):
#         out = self.model(x)
#         if isinstance(out, list):
#             assert len(out) == 1
#             out = out[0]
#         if self.pool:
#             out = self.pool(out).squeeze(-1).squeeze(-1)
#         return out

from safetensors.torch import load_file
class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': False, 'num_classes': 0}, 
                 pool: bool = True,
                 local_checkpoint_path: str = '/sharefiles2/yaoshuilian/resnet50_tv_in1k/model.safetensors'):
        super().__init__()

        assert os.path.exists(local_checkpoint_path), f"本地模型文件不存在：{local_checkpoint_path}"
        assert local_checkpoint_path.endswith('.safetensors'), "文件格式必须为.safetensors"

        self.model = timm.create_model(model_name, **kwargs)
        
        try:
            checkpoint = load_file(local_checkpoint_path, device='cuda')
            # 只保留模型中存在的 key
            model_state_dict = self.model.state_dict()
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
            missing, unexpected = self.model.load_state_dict(filtered_checkpoint, strict=False)
            print(f"Loaded weights from {local_checkpoint_path}")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
        except Exception as e:
            raise Exception(f"加载.safetensors文件失败：{str(e)}") from e
        
        self.model_name = model_name
        self.local_checkpoint_path = local_checkpoint_path
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out