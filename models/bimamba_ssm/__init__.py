__version__ = "1.0.1"
# try:
#     from ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
# except ImportError:
#     selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

# try:
#     from ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None

# try:
#     from ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# from .ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
# from .modules.mamba_simple import Mamba
# from .models.mixer_seq_simple import MambaLMHeadModel

# from modules.mamba_simple import Mamba, Block
# from utils.generation import GenerationMixin
# from utils.hf import load_config_hf, load_state_dict_hf

# try:
#     from ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
