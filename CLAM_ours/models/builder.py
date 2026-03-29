import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
import os
import json
from timm.layers import SwiGLUPacked

os.environ['CONCH_CKPT_PATH'] = 'xxx/conch_pytorch_model.bin'
os.environ['UNI_CKPT_PATH'] = 'xxx/UNI_pytorch_model.bin'

os.environ['GIGA_CKPT_PATH'] = 'xxx/gigapath/pytorch_model.bin'
os.environ['VIRCHOW_CKPT_PATH'] = 'xxx/virchow/pytorch_model.bin'

os.environ['GIGA_CONFIG_PATH'] = 'xxx/gigapath/config.json'
os.environ['VIRCHOW_CONFIG_PATH'] = 'xxx/virchow/config.json'

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
import os
import torch
import json
import timm
from functools import partial

def has_GIGA():
    HAS_GIGA = False
    GIGA_CKPT_PATH = ''
    GIGA_CONFIG_PATH = ''
    try:
        if 'GIGA_CKPT_PATH' not in os.environ:
            raise ValueError('GIGA_CKPT_PATH not set')
        if 'GIGA_CONFIG_PATH' not in os.environ:
            raise ValueError('GIGA_CONFIG_PATH not set')

        if not os.path.exists(os.environ['GIGA_CKPT_PATH']):
            raise FileNotFoundError(f"GigaPath checkpoint not found: {os.environ['GIGA_CKPT_PATH']}")
        if not os.path.exists(os.environ['GIGA_CONFIG_PATH']):
            raise FileNotFoundError(f"GigaPath config not found: {os.environ['GIGA_CONFIG_PATH']}")

        import timm
        HAS_GIGA = True
        GIGA_CKPT_PATH = os.environ['GIGA_CKPT_PATH']
        GIGA_CONFIG_PATH = os.environ['GIGA_CONFIG_PATH']
    except Exception as e:
        print(e)
        print('GigaPath files or dependencies missing')
    return HAS_GIGA, GIGA_CKPT_PATH, GIGA_CONFIG_PATH

def has_VIRCHOW():
    HAS_VIRCHOW = False
    VIRCHOW_CKPT_PATH = ''
    try:
        if 'VIRCHOW_CKPT_PATH' not in os.environ:
            raise ValueError('VIRCHOW_CKPT_PATH not set')

        if not os.path.exists(os.environ['VIRCHOW_CKPT_PATH']):
            raise FileNotFoundError(f"VIRCHOW checkpoint not found: {os.environ['VIRCHOW_CKPT_PATH']}")

        import timm
        assert timm.__version__ >= "0.9.11", f"timm version must be >=0.9.11, current: {timm.__version__}"

        HAS_VIRCHOW = True
        VIRCHOW_CKPT_PATH = os.environ['VIRCHOW_CKPT_PATH']
    except Exception as e:
        print(e)
        print('VIRCHOW files or dependencies missing or incompatible')
    return HAS_VIRCHOW, VIRCHOW_CKPT_PATH

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()

    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5,
                            num_classes=0,
                            dynamic_img_size=True)

        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN requires 448x448 input size'
    elif model_name == 'plip':
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(
            "xxx/plip/"
        )

        def forward_image(pixel_values):
            return model.get_image_features(pixel_values)

        model.forward = forward_image
    elif model_name == 'gigapath':
        HAS_GIGA, GIGA_CKPT_PATH, GIGA_CONFIG_PATH = has_GIGA()
        assert HAS_GIGA, 'GigaPath is not available'

        with open(GIGA_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        model = timm.create_model(
            model_name=config['architecture'],
            checkpoint_path=GIGA_CKPT_PATH,
            **config["model_args"]
        )

        model = model.eval()

    elif model_name == 'VIRCHOW':
        HAS_VIRCHOW, VIRCHOW_CKPT_PATH = has_VIRCHOW()
        assert HAS_VIRCHOW, 'VIRCHOW is not available'

        model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            checkpoint_path=VIRCHOW_CKPT_PATH,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            img_size=224,
            init_values=1e-5,
            num_classes=0,
            mlp_ratio=5.3375,
            global_pool="",
            dynamic_img_size=True,
        )

        model = model.eval()

        def virchow_forward(self, x):
            with torch.autocast(device_type="cuda" if x.is_cuda else "cpu", dtype=torch.float16):
                output = self.forward_features(x)
                class_token = output[:, 0]
                patch_tokens = output[:, 1:]
                embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
                return embedding

        model.forward = virchow_forward.__get__(model, type(model))

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms