# Includes modified LimiX transformer code. Copyright 2025 StableAI.
# FEAT adds sample mixing, cell encoding, and inference optimizations.
# See NOTICE and third_party/limix/README.md for the upstream source and license.
from __future__ import annotations

import torch
import torch.nn as nn
from .layer import EncoderBaseLayer, LayerStack
from typing import Any, Literal, Optional
from .encoders import get_x_encoder, get_cls_y_encoder, get_reg_y_encoder, preprocesss_4_x, NormalizationEncoder
from torch.amp import autocast
import math


from ..utils.profiler import get_profiler


class FEATModel(nn.Module):
    """FEAT architecture with cell-level encoding and checkpoint-based inference."""
    def __init__(
                self,
                *,
                preprocess_config_x:dict[str, Any],
                encoder_config_x:dict[str, Any],
                encoder_config_y:dict[str, Any],
                decoder_config:dict[str, Any],
                nlayers:int,
                nhead: int, 
                embed_dim: int, 
                hid_dim:int,
                feature_positional_embedding_type:Literal['none','subortho'] = 'subortho',
                mask_prediction: bool = False,
                features_per_group:int = 2,
                dropout: float=0,
                pre_norm: bool=False,
                activation: str='gelu',
                layer_norm_eps: float=1e-5,
                device: torch.device|None=None,
                dtype: torch.dtype|None=None,
                recompute_attn: bool=False,
                mlp_use_residual:bool=False,
                layer_arch: str = 'fmfmsm',
                # =================================================================
                # Cell-Level Encoding Configuration (De Novo Training)
                # Supports UNLIMITED sequence lengths via sinusoidal encoding!
                # =================================================================
                use_cell_encoding: bool = True,           # Enable/disable cell positional encoding
                cell_encoding_type: str = 'sinusoidal',   # 'sinusoidal' (unlimited) or 'learned'
                max_cell_rows: Optional[int] = None,      # Only for 'learned' type, None=use sinusoidal
                max_cell_cols: Optional[int] = None,      # Only for 'learned' type, None=use sinusoidal
                cell_embedding_dim: Optional[int] = None, # Defaults to embed_dim
                **layer_kwargs:Any
                ):
        super().__init__()
        
        self.preprocess_config_x = preprocess_config_x
        self.encoder_config_x = encoder_config_x
        self.encoder_config_y = encoder_config_y
        self.decoder_config = decoder_config
        self.feature_positional_embedding_type = feature_positional_embedding_type
        self.nlayers = nlayers
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.mask_prediction = mask_prediction
        self.features_per_group = features_per_group
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.recompute_attn = recompute_attn
        self.mlp_use_residual = mlp_use_residual
        self.layer_arch = layer_arch
        
        # =================================================================
        # Cell-Level Encoding Configuration (De Novo Training)
        # Supports UNLIMITED sequence lengths via sinusoidal encoding!
        # =================================================================
        self.use_cell_encoding = use_cell_encoding
        self.cell_encoding_type = cell_encoding_type
        self.max_cell_rows = max_cell_rows  # Only used if cell_encoding_type='learned'
        self.max_cell_cols = max_cell_cols  # Only used if cell_encoding_type='learned'
        self.cell_embedding_dim = cell_embedding_dim if cell_embedding_dim is not None else embed_dim

        # [MODIFIED] Respect sample_mixer_type from config
        # Default to 'mamba' if not specified
        default_mixer_type = layer_kwargs.pop('sample_mixer_type', 'mamba')
        
        # [MODIFIED] Configurable Mamba/Linear Mixing Ratio
        # Default: 8 (Insert Linear every 8th layer -> 7 Mamba + 1 Linear)
        # Set to 0 to disable Linear layers (Pure Mamba).
        # Set to 1 to use Linear only.
        linear_attn_interval = layer_kwargs.pop('linear_attn_interval', 8)
        
        # [IMPROVEMENT] Support for Hybrid Full Attention injection
        # Important for "retrieval" capabilities in Mamba architectures.
        # Allows inserting standard MultiheadAttention layers periodically.
        hybrid_attn_interval = layer_kwargs.pop('hybrid_attn_interval', 0)

        layers = []
        for i in range(self.nlayers):
            # Customizable hybrid pattern ONLY if default is 'mamba'
            # If user explicitly asks for 'attention' or 'linear', use that uniformly.
            if default_mixer_type == 'mamba':
                if hybrid_attn_interval > 0 and (i + 1) % hybrid_attn_interval == 0:
                    mixer_type = 'attention'
                elif linear_attn_interval > 0 and (i + 1) % linear_attn_interval == 0:
                    mixer_type = 'linear'
                else:
                    mixer_type = 'mamba'
            else:
                mixer_type = default_mixer_type
            
            layers.append(EncoderBaseLayer(
                embed_dim=self.embed_dim,
                hid_dim=self.hid_dim,
                nhead=self.nhead,
                dropout=self.dropout,
                pre_norm=self.pre_norm,
                activation=self.activation, # type: ignore
                layer_norm_eps=self.layer_norm_eps,
                device=self.device,
                dtype=self.dtype,
                recompute_attn=self.recompute_attn,
                mlp_use_residual=self.mlp_use_residual,
                layer_arch=self.layer_arch, # type: ignore
                sample_mixer_type=mixer_type,
                use_checkpoint=True, # Enable gradient checkpointing for Mamba/Linear blocks
                **layer_kwargs
            ))

        self.transformer_encoder = LayerStack(layers)
        self.encoder_out_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=False) if pre_norm else nn.Identity()

        self.cls_y_decoder = nn.Sequential(
                                            nn.Linear(self.embed_dim, self.hid_dim),
                                            nn.GELU(),
                                            nn.Linear(self.hid_dim, decoder_config['num_classes']),
                                            )
        
        self.reg_y_decoder = nn.Sequential(
                                        nn.Linear(self.embed_dim, self.hid_dim),
                                        nn.LayerNorm(self.hid_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dim, 1),
                                        )
        self.feature_decoder = nn.Sequential(
                                        nn.Linear(self.embed_dim, self.hid_dim),
                                        nn.LayerNorm(self.hid_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dim, self.features_per_group),
                                        )
        
        if feature_positional_embedding_type == "learned":
            self.feature_positional_embedding = nn.Embedding(1_000, self.embed_dim)
        elif feature_positional_embedding_type == "subspace":
            self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)
        elif feature_positional_embedding_type == "subortho":
            # =================================================================
            # DFE (Discriminative Feature Encoding) - SubOrtho Initialization
            # =================================================================
            # CRITICAL for Column Permutation Invariance!
            # 
            # Key insight from LimiX paper: Feature embeddings should provide:
            # - Column IDENTITY (which column is this?)
            # - NOT column POSITION (no order/distance implied)
            #
            # Design:
            # 1. LEARNABLE low-rank vectors (not sinusoidal/fixed)
            # 2. ORTHOGONAL initialization for maximum inter-column discrimination
            # 3. NO ordering bias - column i is not "closer" to column i+1
            # 4. Applied to ALL tokens (including MASK) to preserve "which column"
            #
            # Technical Implementation:
            # - Low-rank base: [num_groups, embed_dim // 4] (learnable)
            # - Linear projection: embed_dim // 4 -> embed_dim (learnable)
            # - Result: Each column gets a UNIQUE, LEARNABLE identity vector
            #
            # Why Orthogonal Init?
            # - Maximizes initial discrimination between columns
            # - Prevents collapse where multiple columns get similar embeddings
            # - Orthogonal vectors have cosine similarity = 0 (maximally distinct)
            # =================================================================
            
            # Low-rank to full-rank projection (learnable)
            self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim, bias=False)
            
            # Maximum number of feature groups (columns) to support
            # Can be dynamically expanded if needed (see add_embeddings)
            self._max_feature_groups = 512
            
            # Learnable low-rank column identifier vectors
            # Shape: [max_groups, embed_dim // 4]
            # After projection: [max_groups, embed_dim]
            self._feature_emb_base = nn.Parameter(
                torch.empty(self._max_feature_groups, self.embed_dim // 4, device=device, dtype=dtype)
            )
            # SubOrtho initialization: orthogonal for maximum discrimination
            # Note: Orthogonal != Position. Columns 1 and 100 are equally "far"
            nn.init.orthogonal_(self._feature_emb_base)
        
        self.x_preprocess = preprocesss_4_x(**preprocess_config_x)
        self.encoder_x = get_x_encoder(**encoder_config_x)
        self.cls_y_encoder = get_cls_y_encoder(**encoder_config_y)
        self.reg_y_encoder = get_reg_y_encoder(**encoder_config_y)
        
        # =================================================================
        # CELL-LEVEL ENCODING LAYERS (De Novo Training from Scratch)
        # =================================================================
        # These encodings explicitly model tabular structure:
        # - Row encoding: positional context of each row/sample (sequence position)
        # - Column encoding: positional context of each feature/column
        #
        # We support TWO modes:
        # 1. 'sinusoidal': Unlimited sequence length, no learned parameters
        # 2. 'learned': Fixed max length, uses nn.Embedding (legacy)
        
        self.row_embeddings = None
        self.col_embeddings = None
        
        if self.use_cell_encoding:
            if self.cell_encoding_type == 'learned' and self.max_cell_rows is not None:
                # Legacy: Learned embeddings with fixed max size
                self.row_embeddings = nn.Embedding(
                    num_embeddings=self.max_cell_rows,
                    embedding_dim=self.cell_embedding_dim,
                    device=self.device,
                    dtype=self.dtype
                )
            # For 'sinusoidal', we compute on-the-fly in forward()
            
            if self.cell_encoding_type == 'learned' and self.max_cell_cols is not None:
                # Legacy: Learned embeddings with fixed max size
                self.col_embeddings = nn.Embedding(
                    num_embeddings=self.max_cell_cols,
                    embedding_dim=self.cell_embedding_dim,
                    device=self.device,
                    dtype=self.dtype
                )
            # For 'sinusoidal', we compute on-the-fly in forward()
            
            # LayerNorm after fusing cell embeddings with token embeddings
            self.cell_embedding_norm = nn.LayerNorm(
                self.embed_dim, 
                eps=self.layer_norm_eps,
                device=self.device,
                dtype=self.dtype
            )
        else:
            self.cell_embedding_norm = None
        
        # [FIX] Input Normalization Layer (Pre-Transformer)
        # Ensures that embeddings (especially from unnormalized Regression targets)
        # are normalized before entering the Transformer, preventing NaN/Inf.
        from .layer import LayerNormMixedPrecision
        self.embedding_norm = LayerNormMixedPrecision(self.embed_dim, eps=self.layer_norm_eps, device=self.device, dtype=self.dtype)

        # [OPTIMIZATION] Caches for inference acceleration
        self._cached_feature_embs = {}  # Cache for orthogonal feature embeddings by num_groups
        self._pe_cache = {}  # Cache for sinusoidal positional encodings by (length, dim)

        # =================================================================
        # WEIGHT INITIALIZATION (De Novo Training - Critical for Convergence)
        # =================================================================
        # Since we're training from scratch, proper initialization is crucial
        self._init_weights()
        
        # =================================================================
        # SAFETY CHECK: Ensure all parameters are trainable
        # =================================================================
        self._verify_requires_grad()

    def train(self, mode: bool = True):
        """Override train() to clear inference caches when switching to training mode."""
        if mode:
            self._cached_feature_embs.clear()
            self._pe_cache.clear()
        return super().train(mode)

    def _init_weights(self):
        """
        Custom weight initialization for De Novo Joint Optimization.
        
        Initialize parameters before the checkpoint loader copies trained weights.
        
        Strategy:
        - Linear layers: Xavier uniform initialization (maintains variance)
        - Embedding layers: Normal distribution (std=0.02, following BERT/GPT)
        - LayerNorm: Standard initialization (weight=1, bias=0)
        - Bias terms: Zero initialization
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers - helps with gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                # Normal initialization with std=0.02 for embeddings
                # This follows the BERT/GPT initialization scheme
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Conv1d):
                # Kaiming initialization for convolutional layers (used in Mamba)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for attention output projections
        # Scale down to prevent large initial outputs
        for layer in self.transformer_encoder.layers:
            if hasattr(layer, 'feature_attentions'):
                for attn in layer.feature_attentions:
                    if hasattr(attn, 'out_proj_weight'):
                        # Scale down output projection
                        nn.init.xavier_uniform_(attn.out_proj_weight, gain=1.0 / math.sqrt(2 * self.nlayers))
                        
        # Initialize learned cell embeddings if they exist
        if self.row_embeddings is not None:
            nn.init.normal_(self.row_embeddings.weight, mean=0.0, std=0.02)
        if self.col_embeddings is not None:
            nn.init.normal_(self.col_embeddings.weight, mean=0.0, std=0.02)
            
        print(f"[De Novo Init] Initialized all weights from scratch for joint optimization.")
    
    def _verify_requires_grad(self):
        """
        Safety check to ensure ALL parameters have requires_grad=True.
        
        For De Novo training, we must optimize all parameters jointly.
        This method verifies no parameters were accidentally frozen.
        """
        frozen_params = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params.append(name)
        
        if frozen_params:
            print(f"[WARNING] The following parameters have requires_grad=False:")
            for name in frozen_params[:10]:  # Show first 10
                print(f"  - {name}")
            if len(frozen_params) > 10:
                print(f"  ... and {len(frozen_params) - 10} more")
            print(f"[ACTION] Unfreezing all parameters for De Novo training...")
            for name, param in self.named_parameters():
                param.requires_grad = True
        
        print(f"[De Novo Check] Total params: {total_params:,} | Trainable: {trainable_params:,} | All trainable: {len(frozen_params) == 0}")
    
    def _sinusoidal_positional_encoding(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding for UNLIMITED sequence lengths.

        [IMPROVED] Uses adaptive base frequency scaling for long sequences.
        [OPTIMIZED] Caches results by (length, dim) during inference to avoid recomputation.
        """
        # [OPTIMIZATION] Check cache during inference
        cache_key = (length, dim)
        if not self.training and cache_key in self._pe_cache:
            cached = self._pe_cache[cache_key]
            if cached.device == device and cached.dtype == dtype:
                return cached
            return cached.to(device=device, dtype=dtype)

        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)

        base_freq = 10000.0
        if length > base_freq:
            scale = length / base_freq
            base_freq = base_freq * (scale ** (dim / (dim - 2)))

        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) *
                            (-math.log(base_freq) / dim))

        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2] if dim % 2 == 1 else div_term)

        result = pe.to(dtype)

        # Cache during inference
        if not self.training:
            self._pe_cache[cache_key] = result

        return result
    
    def _get_cell_encoding(self, seq_len: int, num_groups: int, device: torch.device, dtype: torch.dtype):
        """
        Get cell-level positional encodings for rows and columns.
        
        =================================================================
        COLUMN PERMUTATION INVARIANCE DESIGN
        =================================================================
        For tabular data, columns (features) have NO inherent order.
        Using sinusoidal positional encoding for columns would introduce
        a FALSE inductive bias that "column 1 is near column 2".
        
        Solution:
        - ROW encoding: Use sinusoidal PE (rows may have temporal order)
        - COLUMN encoding: Return ZEROS here, rely ENTIRELY on 
          Discriminative Feature Encoding (DFE/SubOrtho) in add_embeddings()
          
        DFE provides column IDENTITY (which column is this?) without
        implying any spatial/ordering relationship between columns.
        This preserves the column permutation invariance property of tables.
        =================================================================
        
        Args:
            seq_len: Number of samples/rows (can be ANY value, e.g., 100000+)
            num_groups: Number of feature groups/columns (can be ANY value)
            device: Target device
            dtype: Target dtype
            
        Returns:
            row_encoding: [seq_len, embed_dim] - Sinusoidal for row position
            col_encoding: [num_groups, embed_dim] - ZEROS (column identity via DFE)
        """
        if self.cell_encoding_type == 'sinusoidal':
            # Row encoding: Sinusoidal PE for sample/row position
            # (Useful if rows have temporal meaning, e.g., time series)
            row_encoding = self._sinusoidal_positional_encoding(
                seq_len, self.cell_embedding_dim, device, dtype
            )
            
            # =============================================================
            # CRITICAL FIX: Column encoding is ZEROS
            # =============================================================
            # We do NOT use sinusoidal PE for columns because:
            # 1. Tables have column PERMUTATION INVARIANCE
            # 2. Sinusoidal PE implies "distance" between columns (wrong!)
            # 3. Column identity is provided by DFE (SubOrtho) in add_embeddings()
            # =============================================================
            col_encoding = torch.zeros(
                num_groups, self.cell_embedding_dim, device=device, dtype=dtype
            )
        else:
            # Learned embeddings (legacy, with max limit)
            if self.row_embeddings is not None:
                row_ids = torch.arange(seq_len, device=device).clamp(max=self.max_cell_rows - 1)
                row_encoding = self.row_embeddings(row_ids)
            else:
                row_encoding = torch.zeros(seq_len, self.cell_embedding_dim, device=device, dtype=dtype)
            
            # For columns: ZEROS even in 'learned' mode
            # Column identity is handled by DFE (SubOrtho) in add_embeddings()
            col_encoding = torch.zeros(num_groups, self.cell_embedding_dim, device=device, dtype=dtype)
        
        return row_encoding, col_encoding

    def forward(self, x: torch.Tensor, 
                y: torch.Tensor, 
                eval_pos: int, 
                y_type: torch.Tensor = None,
                task_type: Literal['reg', 'cls'] = 'cls',
                calculate_sample_attention: bool = False,
                calculate_feature_attention: bool = False,
                return_embedding: bool = False,
                **kwargs
                ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        '''
        Forward pass with Cell-Level Encoding for tabular structure modeling.
        
        Args:
            x: Input features [batch, sequence, feature]
            y: Labels [batch, label]
            eval_pos: Context/target split point
            y_type: Task type tensor (0=cls, 1=reg)
            task_type: 'cls' or 'reg'
            cell_row_ids: Row position IDs for each sample [batch, seq_len]
                         If None, auto-generated as sequential indices
            cell_col_ids: Column position IDs for each feature/group [batch, num_groups]
                         If None, auto-generated as sequential indices
            cell_span_ids: Optional span IDs for merged cells [batch, seq_len, num_groups]
        
        Returns:
            Model output (predictions or embeddings)
        '''
        assert x is not None and y is not None, "x and y must not be none"
        assert len(x.shape)==3, "x must be [Batch, seq, Feature]"
        # assert len(y.shape)==2, "y must be [Batch, label]"
        # [MODIFIED] LimiX supports [Batch, Seq] or [Batch, Seq, Feature] for y

        assert eval_pos <= x.shape[1] and eval_pos <= y.shape[1], "The split point between train x and test x must be less than or equal to the feature dimension of x, and less than or equal to the label dimension of y"

        _profiler = get_profiler()
        _profiler.start("forward_total")

        batch_size, seq_len, num_feature = x.shape
        _profiler.mark("input_reshape_pad")
        x = {'data':x, 'mask':torch.isnan(x).to(torch.int32).to(x.device)}
        
        # Ensure y is float to support NaN injection for masking
        if not y.is_floating_point():
            y = y.float()
        y = {'data':y}
        
        remainder = num_feature % self.features_per_group
        feature_to_add = 0
        if remainder > 0:
            feature_to_add = self.features_per_group - remainder
            
        if feature_to_add > 0:
            # Extend the feature dimension of x when it is insufficient
            for k in x:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            batch_size,
                            seq_len,
                            feature_to_add,
                            device=x[k].device,
                            dtype=x[k].dtype
                        )
                    ),
                    dim=-1
                )
        
        # Calculate number of feature groups BEFORE reshaping
        num_groups = x['data'].shape[2] // self.features_per_group
        
        for k in x:
            x[k] = x[k].reshape(batch_size, seq_len, x[k].shape[2]//self.features_per_group, self.features_per_group)
        x['eval_pos'] = eval_pos
        
        _profiler.mark("x_dict_build")
        # [FIX] Run preprocessing steps MANUALLY to capture normalized_x at the right point
        # x_preprocess is: NanEncoder -> [ValidFeatureEncoder(nan)] -> NormalizationEncoder -> ValidFeatureEncoder(data)
        # We need to capture data RIGHT AFTER NormalizationEncoder, before the final ValidFeatureEncoder scales it
        preprocessed_x = x
        normalized_x_for_loss = None
        for i, step in enumerate(self.x_preprocess):
            preprocessed_x = step(preprocessed_x)
            # [OPTIMIZATION] Only capture normalized data during training (needed for loss computation)
            if self.training and isinstance(step, NormalizationEncoder):
                normalized_x_for_loss = preprocessed_x['data'].clone()

        # Safety check: ensure we captured the normalized data (training only)
        if self.training and normalized_x_for_loss is None:
            print("[WARN] NormalizationEncoder not found in x_preprocess, using final preprocessed_x")
            normalized_x_for_loss = preprocessed_x['data'].clone()
        
        _profiler.mark("x_preprocess")
        preprocessed_x = self.process_4_x(preprocessed_x)
        x_encoder_result = self.encoder_x(preprocessed_x)
        x_emb_result = x_encoder_result['data']
        _profiler.mark("x_encoding")
        
        # =================================================================
        # CELL-LEVEL ENCODING INJECTION (De Novo Training)
        # =================================================================
        # This is where we inject tabular structure information into the
        # token embeddings. Supports UNLIMITED sequence lengths!
        # 
        # x_emb_result shape: [batch, seq_len, num_groups, embed_dim]
        # We add:
        #   - Row encoding: Position of each sample (row in table)
        #   - Column encoding: Position of each feature group (column in table)
        
        if self.use_cell_encoding:
            actual_num_groups = x_emb_result.shape[2]  # Use actual shape after encoding
            
            # Get cell encodings (sinusoidal or learned)
            # This supports ANY seq_len and num_groups - NO LIMITS!
            row_encoding, col_encoding = self._get_cell_encoding(
                seq_len=seq_len,
                num_groups=actual_num_groups,
                device=x_emb_result.device,
                dtype=x_emb_result.dtype
            )
            
            # Expand for broadcasting to x_emb_result shape [B, S, G, E]
            # Row encoding: [S, E] -> [1, S, 1, E] (broadcast across batch and groups)
            row_encoding = row_encoding.unsqueeze(0).unsqueeze(2)  # [1, S, 1, E]
            
            # Column encoding: [G, E] -> [1, 1, G, E] (broadcast across batch and sequence)
            col_encoding = col_encoding.unsqueeze(0).unsqueeze(1)  # [1, 1, G, E]
            
            # Element-wise addition of cell encodings to token embeddings
            # This fuses tabular structure (row/col position) with semantic content
            x_emb_result = x_emb_result + row_encoding + col_encoding
            
            # Apply LayerNorm to stabilize the fused representation
            if self.cell_embedding_norm is not None:
                x_emb_result = self.cell_embedding_norm(x_emb_result)
        
        # =================================================================
        # END: Cell-Level Encoding Injection
        # =================================================================
        _profiler.mark("cell_encoding")

        for k in y:
            # Extend the label dimension of y when it is insufficient
            y[k] = y[k].unsqueeze(-1)
            if y[k].shape[1] < x['data'].shape[1]:
                y[k] = torch.cat(
                    (
                        y[k],
                        torch.tensor(float('nan'), dtype=y[k].dtype, device=y[k].device)
                        * torch.zeros(
                            y[k].shape[0],
                            x["data"].shape[1] - y[k].shape[1],
                            y[k].shape[2],
                            device=y[k].device,
                            dtype=y[k].dtype,
                        ),
                    ),
                    dim=1
                )
        # Mask the test y
        y["data"][:, eval_pos:] = torch.tensor(float('nan'), dtype=y['data'].dtype, device=y['data'].device)
        
        # [OPTIMIZATION] During inference with uniform task type, skip y_type tensor creation
        # and use fast-path in mixed_y_embedding that avoids gather/scatter indexing
        if not self.training and y_type is None:
            embedded_y = self.mixed_y_embedding(y, y_type=None, eval_pos=eval_pos, task_type=task_type)
        else:
            if y_type is None:
                if task_type == 'cls':
                    y_type = torch.zeros_like(y['data'], device=y['data'].device)
                else:
                    y_type = torch.ones_like(y['data'], device=y['data'].device)
            embedded_y = self.mixed_y_embedding(y, y_type=y_type, eval_pos=eval_pos)

        _profiler.mark("y_embedding")
        if self.training and torch.isnan(embedded_y).any():
            raise ValueError("embedded_y contains NaN values; please add a NanEncoder in the encoder")

        embedded_x = self.add_embeddings(x_emb_result)
        
        # [FIX] Ensure embedded_y matches embedded_x dtype (e.g. BFloat16 vs Float32)
        if embedded_y.dtype != embedded_x.dtype:
            embedded_y = embedded_y.to(embedded_x.dtype)
            
        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        if self.training and torch.isnan(embedded_all).any():
            raise ValueError("embedded_all contains NaN values; please add a NanEncoder in the encoder")
        
        # [FIX] Apply Pre-Transformer Normalization
        # This squashes large values (e.g. from Regression targets) into a stable range.
        embedded_all = self.embedding_norm(embedded_all)
        _profiler.mark("feature_emb_concat_norm")

        if calculate_sample_attention or calculate_feature_attention:
            return self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos,
                                            calculate_sample_attention=calculate_sample_attention,
                                            calculate_feature_attention=calculate_feature_attention, **kwargs)
        else:
            pass
        encoder_out = self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos, **kwargs)[0]
        encoder_out = self.encoder_out_norm(encoder_out)
        _profiler.mark("transformer_encoder")

        test_encoder_out = encoder_out[:, eval_pos:, -1]
        
        if return_embedding:
            return test_encoder_out

        # [OPTIMIZATION] Lazily create y_type for y_decoder if it was skipped during inference
        if y_type is None:
            if task_type == 'cls':
                y_type = torch.zeros(batch_size, seq_len, 1, device=test_encoder_out.device, dtype=test_encoder_out.dtype)
            else:
                y_type = torch.ones(batch_size, seq_len, 1, device=test_encoder_out.device, dtype=test_encoder_out.dtype)
        test_y_type = y_type[:,eval_pos:]
        encoder_out_4_feature = encoder_out[:, :, :-1, :]
        if self.mask_prediction:
            cls_output, reg_output = self.y_decoder(test_encoder_out, test_y_type)
            feature_pred = self.feature_decoder(encoder_out_4_feature)
            _profiler.mark("decoder")
            _profiler.stop()
            
            # [FIX] Select primary prediction for loss calculation
            y_pred = cls_output if task_type == 'cls' else reg_output
            
            output_decoded = {
                "cls_output": cls_output,
                "reg_output": reg_output,
                "y_pred": y_pred, # [FIX] Added missing key
                "feature_pred": feature_pred,
                "process_config": {
                    "n_x_padding": feature_to_add,
                    "features_per_group": self.x_preprocess[3].num_features,
                    "num_used_features": self.x_preprocess[3].valid_feature_num,
                    "mean_for_normalization": self.x_preprocess[2].mean,
                    "std_for_normalization": self.x_preprocess[2].std,
                    "outlier_lower": self.x_preprocess[2].lower,
                    "outlier_upper": self.x_preprocess[2].upper,
                    # [NEW] Provide the normalized x directly as ground truth for loss computation
                    # This is CRITICAL: train.py should use this instead of trying to normalize original_x
                    "normalized_x": normalized_x_for_loss  # [B, S, G, F_per_group] - already normalized!
                }
            }
        else:
            cls_output, reg_output = self.y_decoder(test_encoder_out, test_y_type)
            _profiler.mark("decoder")
            _profiler.stop()
            if task_type=="cls":
                output_decoded = cls_output
            else:
                output_decoded = reg_output

        return output_decoded

    def mixed_y_embedding(self, y:dict, y_type:torch.Tensor, eval_pos:int, task_type:str=None):
        y = y['data']
        # [FIX] Handle flexible y shapes: [Batch, Seq], [Batch, Seq, Feature], or [Batch, Seq, Feature, 1]

        # Squeeze the last implicit dimension if it exists and is 1
        if y.dim() == 4 and y.shape[-1] == 1:
             y = y.squeeze(-1)

        if y.dim() == 2:
             batch_size, seq_len = y.shape
             y_num = 1
             y = y.unsqueeze(-1) # Ensure [Batch, Seq, 1] for processing
        elif y.dim() == 3:
             batch_size, seq_len, y_num = y.shape
        else:
             raise ValueError(f"Unexpected y shape: {y.shape}")

        # [OPTIMIZATION] Fast path for uniform task type during inference.
        # Skips: torch.arange, boolean masking, index select, torch.empty, index_put_ scatter.
        # Safe because during inference, task_type is always uniform (all cls or all reg).
        if task_type is not None and y_type is None:
            y_input = {'data': y, 'eval_pos': eval_pos}
            if task_type == 'cls':
                emb = self.cls_y_encoder(y_input)['data']
            else:
                emb = self.reg_y_encoder(y_input)['data']
            # Encoder output: [B, S, y_num, emb_dim] -> squeeze y_num dim
            if emb.dim() == 4 and emb.shape[2] == 1:
                emb = emb.squeeze(2)  # [B, S, emb_dim]
            elif emb.dim() == 4:
                emb = emb.reshape(batch_size, seq_len, -1)[:, :, :self.embed_dim]
            return emb

        # Original path for mixed task types (training with mixed batches)
        y_flat = y.reshape(-1)
        y_type_flat = y_type.reshape(-1)

        idx = torch.arange(len(y_flat), device=y.device)
        idx_cls = idx[y_type_flat == 0]
        idx_reg = idx[y_type_flat == 1]
        y_cls = y_flat[idx_cls]
        y_reg = y_flat[idx_reg]

        y_cls = y_cls.reshape(-1, seq_len, y_num)
        y_reg = y_reg.reshape(-1, seq_len, y_num)
        y_cls = {'data': y_cls, 'eval_pos':eval_pos}
        y_reg = {'data': y_reg, 'eval_pos':eval_pos}

        cls_y_emb = self.cls_y_encoder(y_cls) if len(idx_cls) > 0 else None
        reg_y_emb = self.reg_y_encoder(y_reg) if len(idx_reg) > 0 else None
        cls_y_emb = cls_y_emb['data'] if cls_y_emb is not None else None
        reg_y_emb = reg_y_emb['data'] if reg_y_emb is not None else None

        emb_size = self.embed_dim
        out = torch.empty(len(y_flat), emb_size, dtype=y.dtype, device=y.device)

        if cls_y_emb is not None:
            cls_y_emb_flat = cls_y_emb.reshape(-1, emb_size).to(dtype=out.dtype)
            out.index_put_((idx_cls,), cls_y_emb_flat)

        if reg_y_emb is not None:
            reg_y_emb_flat = reg_y_emb.reshape(-1, emb_size).to(dtype=out.dtype)
            out.index_put_((idx_reg,), reg_y_emb_flat)

        output = out.reshape(batch_size, seq_len, emb_size)
        return output
    
    def process_4_x(self, data:dict):
        x_input = data['data']
        mask = data['mask'].to(torch.bool)
        
        # Ensure mask shape matches x_input shape
        # x_input might have been reshaped/padded in forward() before x_preprocess
        # mask should have undergone the same transformations in forward()
        
        if mask.shape != x_input.shape:
            # If shapes don't match, it's likely due to the reshaping in forward()
            # x[k] = x[k].reshape(batch_size, seq_len, x[k].shape[2]//self.features_per_group, self.features_per_group)
            # We need to ensure mask is compatible.
            # However, in forward(), we iterate over keys in x (which includes 'mask') and apply the same operations.
            # So theoretically they should match.
            # The error "tensor a (2) must match tensor b (16)" suggests a mismatch in the last dimension.
            # This usually happens if 'mask' wasn't updated correctly or if x_preprocess changed x_input's shape but not mask's.
            
            # Let's look at ValidFeatureEncoder in encoders.py. It pads 'data' to num_features.
            # But it doesn't seem to pad other keys like 'mask'.
            # So x_input (data['data']) got padded/transformed, but data['mask'] didn't.
            
            # We need to align mask to x_input.
            # Since x_input might be padded with zeros (which are not NaNs), we can pad mask with False.
            
            if mask.dim() == x_input.dim():
                 # Try to pad mask to match x_input
                 pad_dims = []
                 for i in range(x_input.dim() - 1, -1, -1):
                     diff = x_input.shape[i] - mask.shape[i]
                     pad_dims.extend([0, diff]) # Pad end, pad start
                 
                 if any(p < 0 for p in pad_dims):
                      # If mask is larger, we might need to slice? Or it's a different issue.
                      # For now, let's assume it's padding.
                      pass
                 else:
                      mask = torch.nn.functional.pad(mask, pad_dims, value=False)
            
        x_input = torch.where(mask, float('nan'), x_input)
        data['data'] = x_input
        return data
    
    def add_embeddings(self, x:torch.Tensor):
        """
        Add Discriminative Feature Encoding (DFE) to token embeddings.
        
        =================================================================
        DFE: COLUMN IDENTITY, NOT COLUMN POSITION
        =================================================================
        
        WHY DFE (not Sinusoidal PE for columns)?
        ----------------------------------------
        Tabular data has COLUMN PERMUTATION INVARIANCE:
        - [Age, Income, Education] == [Education, Age, Income]
        - Columns have no inherent order or distance relationship
        
        Sinusoidal PE would wrongly tell the model:
        - "Column 1 is near Column 2, far from Column 100"
        - This is FALSE for tables!
        
        DFE solves this by providing:
        - IDENTITY: "This is the 'Age' column" (unique signature)
        - NOT POSITION: No "Column i is near Column i+1" bias
        
        TECHNICAL DESIGN (LimiX Paper):
        -------------------------------
        1. Low-Rank Learnable Base: [num_columns, embed_dim // 4]
           - Each column has a learnable "ID card"
           - Orthogonal initialization for maximum discrimination
           
        2. Linear Projection: embed_dim // 4 -> embed_dim
           - Allows model to learn optimal representation
           
        3. ADD to All Tokens: Including MASK tokens
           - Even when value is masked, model knows WHICH column
           - Critical for CCMM pretraining (predict masked values)
        
        WHY ORTHOGONAL INIT?
        --------------------
        - Orthogonal vectors have cosine_similarity = 0
        - Column 1 is equally "distant" from Column 2 and Column 100
        - No artificial ordering imposed on columns
        - Maximum discrimination power at initialization
        
        Args:
            x: Token embeddings [Batch, Seq, Groups, Embed_dim]
            
        Returns:
            x + DFE: Token embeddings with column IDENTITY (not position)
        =================================================================
        """
        if self.feature_positional_embedding_type == "subortho":
            num_groups = x.shape[2]
            cache_key = num_groups
            if self.training or cache_key not in self._cached_feature_embs:
                # Generate fresh orthogonal embeddings (always during training, once per shape during inference)
                with autocast(device_type=x.device.type, enabled=False):
                    embs = torch.randn(
                        (num_groups, x.shape[3] // 4),
                        device=x.device,
                        dtype=torch.float32,
                    )
                    torch.nn.init.orthogonal_(embs)
                embs = self.feature_positional_embedding(embs.to(x.dtype))
                if not self.training:
                    self._cached_feature_embs[cache_key] = embs.detach()
            else:
                embs = self._cached_feature_embs[cache_key]
                if embs.device != x.device or embs.dtype != x.dtype:
                    embs = embs.to(device=x.device, dtype=x.dtype)
                    self._cached_feature_embs[cache_key] = embs
            x += embs[None, None]
            
        elif self.feature_positional_embedding_type == "learned":
            # Legacy: Use nn.Embedding lookup
            num_groups = x.shape[2]
            group_ids = torch.arange(num_groups, device=x.device)
            feature_embs = self.feature_positional_embedding(group_ids)  # [num_groups, embed_dim]
            x = x + feature_embs[None, None, :, :]
            
        elif self.feature_positional_embedding_type is None or self.feature_positional_embedding_type == "none":
            pass  # No feature embedding
        else:
            raise ValueError(f"Unknown feature_positional_embedding_type={self.feature_positional_embedding_type}")
        
        return x
    
    def y_decoder(self, test_encoder_out, test_y_type):
        _, seq_len, emb_size = test_encoder_out.shape
        flat_test_encoder_out = test_encoder_out.reshape(-1, emb_size)
        flat_test_y_type = test_y_type.reshape(-1)
        
        idx = torch.arange(len(flat_test_encoder_out), device=flat_test_encoder_out.device)
        idx_cls = idx[flat_test_y_type == 0]
        idx_reg = idx[flat_test_y_type == 1]

        cls_y_encoder_out = flat_test_encoder_out[idx_cls]
        reg_y_encoder_out = flat_test_encoder_out[idx_reg]
        
        if seq_len > 0:
            cls_y_encoder_out = cls_y_encoder_out.reshape(-1, seq_len, emb_size)
            reg_y_encoder_out = reg_y_encoder_out.reshape(-1, seq_len, emb_size)

        cls_y = self.cls_y_decoder(cls_y_encoder_out)
        reg_y = self.reg_y_decoder(reg_y_encoder_out)


        return cls_y, reg_y
