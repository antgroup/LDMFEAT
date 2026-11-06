# Adapted from LimiX model loading. Copyright 2025 StableAI.
# FEAT changes checkpoint validation, model configuration, and inference setup.
# See NOTICE and third_party/limix/README.md for the upstream source and license.
import torch

from feat.model.feat_model import FEATModel
from feat.model.mamba_layer import Mamba2

def build_model(config:dict):
    if config.get('sample_mixer_type', 'mamba') == 'mamba' and Mamba2 is None:
        raise ImportError(
            "FEAT-38M requires Mamba2 from mamba-ssm. Follow the CUDA installation "
            "instructions in README.md before loading the checkpoint."
        )
    model = FEATModel(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        feature_positional_embedding_type=config.get('feature_positional_embedding_type', "subortho"),
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=config.get('mask_prediction', False),
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        pre_norm=config.get('pre_norm', True),
        activation=config.get('activation', 'gelu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        mlp_use_residual=config.get('mlp_use_residual', False),
        use_cell_encoding=config.get('use_cell_encoding', True),
        cell_encoding_type=config.get('cell_encoding_type', 'sinusoidal'),
        max_cell_rows=config.get('max_cell_rows', None),
        max_cell_cols=config.get('max_cell_cols', None),
        cell_embedding_dim=config.get('cell_embedding_dim', None),
        device=config.get('device', None),
        dtype=config.get('dtype', None),
        recompute_attn=config['recompute_attn'],
        layer_arch=config.get('layer_arch', 'fmfmsm'),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
        # Pass Mixing Parameters from Config
        linear_attn_interval=config.get('linear_attn_interval', 8),
        hybrid_attn_interval=config.get('hybrid_attn_interval', 0),
        sample_mixer_type=config.get('sample_mixer_type', 'mamba'),
        d_state=config.get('d_state', 128),
    )
    return model

def load_model(model_path, mask_prediction: bool = False, compile_model: bool = False):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    required_keys = {'config', 'state_dict'}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"Invalid FEAT checkpoint; missing keys: {sorted(missing_keys)}")
    config = dict(checkpoint['config'])
    config['mask_prediction'] = mask_prediction

    model = build_model(config)
    model.load_state_dict(checkpoint['state_dict'])

    # Release the checkpoint container after copying model weights.
    del checkpoint

    # Freeze parameters for inference safety and set to eval mode
    for param in model.parameters():
        param.requires_grad = False

    print(f"[FEAT] Successfully loaded pretrained weights from {model_path}")
    print("[FEAT] Model set to eval mode and parameters frozen.")

    model.eval()

    # [OPTIMIZATION] Optional torch.compile for inference acceleration
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[FEAT] Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"[FEAT] torch.compile failed ({e}), using eager mode")

    return model
