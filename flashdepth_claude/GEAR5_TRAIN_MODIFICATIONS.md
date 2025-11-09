# Critical Modifications Needed for train_gear5.py

## ✅ Completed (via modify_train_gear5.py)
- [x] Renamed `self.phase` → `self.step` throughout
- [x] Updated project names `gear3` → `gear5`
- [x] Updated checkpoint paths and logging

## 🔧 Critical Remaining Modifications

### 1. Update Hydra config path (Line ~1504)
```python
# BEFORE:
@hydra.main(version_base=None, config_path="configs/gear3", config_name="config")

# AFTER:
@hydra.main(version_base=None, config_path="configs/gear5", config_name="config")
```

### 2. Replace _setup_model method (Lines ~232-427)
Key changes:
- **Step 1**: Add `GlobalScalePredictorMultiLayer` instead of `Gear3UpgradeMetricHead`
- **Step 2**: Add `Gear5MetricHead` (includes GSP + FG modulation)
- Enable attention storage for layers [4, 11, 17, 23] (same for both CLS and attention)
- Load Step 1 checkpoint for Step 2 (not hybrid/gear3 checkpoint)

```python
def _setup_model(self):
    """Initialize FlashDepth with Gear5 modules"""
    model = FlashDepth(**model_config)

    embed_dim = 1024 if model.encoder == 'vitl' else 384
    dpt_dim = 256 if model.encoder == 'vitl' else 64

    # Load pretrained FlashDepth weights (ViT + DPT)
    checkpoint_path = self.config.get('load')  # Same for both steps
    if checkpoint_path:
        # Load ViT + DPT only
        # Exclude: mamba, output_conv, gear5_head, global_gsp, fg_modulation_head

    # Add Gear5 modules based on step
    if self.step == 1:
        # Step 1: Add Global GSP only
        model.global_gsp = GlobalScalePredictorMultiLayer(
            embed_dim=embed_dim, num_layers=4
        )
        # Mamba + Final head will train from FlashDepth pretrained

    else:  # self.step == 2
        # Step 2: Load Step 1 checkpoint first
        step1_checkpoint_path = self.config.get('step1_checkpoint')
        if not step1_checkpoint_path:
            raise ValueError("Step 2 requires 'step1_checkpoint' in config!")

        # Load Step 1 checkpoint (includes trained GSP)
        checkpoint = torch.load(step1_checkpoint_path)
        model.load_state_dict(checkpoint['model'], strict=False)

        # Add Gear5MetricHead (includes frozen GSP + trainable FG modulation)
        model.gear5_metric_head = Gear5MetricHead(
            embed_dim=embed_dim, dpt_dim=dpt_dim
        )

        # Copy trained GSP weights from Step 1
        model.gear5_metric_head.global_gsp.load_state_dict(
            checkpoint['model']['global_gsp']
        )

    # Enable attention storage for layers [4, 11, 17, 23]
    # Used for: CLS extraction (Step 1) and multi-layer attention (Step 2)
    encoder_layers = {
        'vitl': [4, 11, 17, 23],  # Encoder output layers
        'vits': [? human: please check]  # TODO: Check ViT-S encoder layers
    }
    target_blocks = encoder_layers[model.encoder]

    for i, block in enumerate(model.pretrained.blocks):
        block.attn.store_attn_weights = (i in target_blocks)

    self.target_blocks = target_blocks
    model = model.to(self.device)

    # Wrap with DDP...
    return model
```

### 3. Update _configure_parameters method (Lines ~429-490)
```python
def _configure_parameters(self, model):
    """Configure trainable/frozen parameters based on step"""

    for name, param in model.named_parameters():
        if self.step == 1:
            # Step 1: Trainable = GSP + Mamba + Final head
            if any(x in name for x in ['global_gsp', 'mamba', 'output_conv']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        elif self.step == 2:
            # Step 2: Trainable = FG modulation + Mamba + Final head
            #         Frozen = ViT + DPT + GSP
            if 'fg_modulation_head' in name:
                param.requires_grad = True
            elif 'global_gsp' in name:
                param.requires_grad = False  # Freeze GSP
            elif any(x in name for x in ['mamba', 'output_conv']):
                param.requires_grad = True  # Continue training
            else:
                param.requires_grad = False  # Freeze ViT + DPT
```

### 4. Update _set_train_mode method (Lines ~477-492)
```python
def _set_train_mode(self):
    """Set trainable parts to train mode, frozen parts to eval mode"""
    self.model.train()

    for name, module in self.model.named_modules():
        if name == '':
            continue

        # Step 1: Keep GSP, mamba, output_conv in train mode
        if self.step == 1:
            if any(kw in name for kw in ['global_gsp', 'mamba', 'output_conv']):
                continue

        # Step 2: Keep FG modulation, mamba, output_conv in train mode
        elif self.step == 2:
            if any(kw in name for kw in ['fg_modulation_head', 'mamba', 'output_conv']):
                continue
            if 'global_gsp' in name:
                module.eval()  # Explicitly freeze GSP
                continue

        # Set frozen parts to eval mode
        module.eval()
```

### 5. Update train_step method (Lines ~903-1000)
Key changes in forward pass:

```python
def train_step(self, batch):
    """Training step with step-specific forward pass"""
    # ... (batch unpacking same)

    # Extract encoder features
    encoder_features = model.pretrained.get_intermediate_layers(
        images_flat, [4, 11, 17, 23]  # Encoder output layers
    )

    # Extract CLS tokens from multiple layers
    cls_tokens_list = [
        encoder_features[0][:, 0],  # Layer 4
        encoder_features[1][:, 0],  # Layer 11
        encoder_features[2][:, 0],  # Layer 17
        encoder_features[3][:, 0],  # Layer 23
    ]

    # Get DPT features (frozen)
    dpt_features = model.depth_head.get_forward_features(
        encoder_features, patch_h, patch_w
    )
    path_1 = dpt_features[-1]  # [B*T, dpt_dim, h, w]

    if self.step == 1:
        # ===== Step 1: Global GSP modulation =====
        # Predict global scale and shift
        scale, shift = model.global_gsp(cls_tokens_list)  # [B]

        # Apply to DPT features
        scale_4d = scale.view(B*T, 1, 1, 1)
        shift_4d = shift.view(B*T, 1, 1, 1)
        path_1_global = path_1 * scale_4d + shift_4d

        # Pass through Mamba + Final head
        path_1_temporal = model.dpt_features_to_mamba(..., path_1_global, ...)
        out = model.depth_head.scratch.output_conv1(path_1_temporal)
        out = F.interpolate(out, (H, W), mode="bilinear")
        out = model.depth_head.scratch.output_conv2(out)

        # Loss on gt valid & pred inlier pixels (same as gear3)

    elif self.step == 2:
        # ===== Step 2: Global + FG modulation =====
        # Collect multi-layer attention [4, 11, 17, 23]
        attention_weights_multi_layer = [
            model.pretrained.blocks[i].attn.attn_weights
            for i in self.target_blocks  # [4, 11, 17, 23]
        ]

        patch_tokens = encoder_features[-1]  # Layer 23 with CLS

        # Forward through Gear5MetricHead
        outputs = model.gear5_metric_head(
            cls_tokens_list=cls_tokens_list,
            patch_tokens=patch_tokens,
            attention_weights_multi_layer=attention_weights_multi_layer,
            dpt_features=path_1,
            patch_h=patch_h,
            patch_w=patch_w,
            step=2
        )

        path_1_fg_modulated = outputs['modulated_features']
        fg_mask = outputs['fg_mask']

        # Pass through Mamba + Final head
        path_1_temporal = model.dpt_features_to_mamba(..., path_1_fg_modulated, ...)
        out = model.depth_head.scratch.output_conv1(path_1_temporal)
        out = F.interpolate(out, (H, W), mode="bilinear")
        out = model.depth_head.scratch.output_conv2(out)

        # Loss on gt valid & pred inlier & FG pixels
        # Upsample fg_mask to match output size
        fg_mask_up = F.interpolate(fg_mask, (H, W), mode='bilinear')
        valid_mask = valid_mask & (fg_mask_up.squeeze(1) > 0.5)
```

### 6. Update validation_step method (Lines ~1002-1100)
- Same forward pass logic as train_step
- Step 1: Use global_gsp
- Step 2: Use gear5_metric_head with step=2

### 7. Update _setup_optimizer method
```python
def _setup_optimizer(self):
    """Setup step-specific optimizer"""
    if self.step == 1:
        # Step 1: GSP (1e-4) + Mamba (1e-4) + Final (1e-4)
        param_groups = [
            {'params': gsp_params, 'lr': 1e-4, 'name': 'gsp'},
            {'params': mamba_params, 'lr': 1e-4, 'name': 'mamba'},
            {'params': output_params, 'lr': 1e-4, 'name': 'output'}
        ]
    elif self.step == 2:
        # Step 2: FG modulation (1e-4) + Mamba (5e-5, lower) + Final (5e-5)
        param_groups = [
            {'params': fg_mod_params, 'lr': 1e-4, 'name': 'fg_mod'},
            {'params': mamba_params, 'lr': 5e-5, 'name': 'mamba'},
            {'params': output_params, 'lr': 5e-5, 'name': 'output'}
        ]
```

### 8. Update _setup_data_loaders method
- Step 1: Use 518×518, 5 datasets
- Step 2: Use 518×518 (NOT 2K for now), same or different datasets
- Remove Phase 2/3 logic (no hybrid training in gear5)

### 9. Update save_checkpoint method (Line ~1480)
```python
checkpoint = {
    'model': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'scheduler': self.scheduler.state_dict(),
    'step': self.step,  # Save which step this is from
    'global_step': self.global_step,
    # ...
}
```

## 📝 Additional Notes

### ViT-S Encoder Layers
Need to verify encoder output layers for ViT-S:
- ViT-L (24 blocks): [4, 11, 17, 23] confirmed
- ViT-S (12 blocks): [?, ?, ?, ?] - need to check

Proportional mapping: ~17%, 46%, 71%, 96%
- ViT-S: 12 blocks → [2, 5, 8, 11] approximately

### Testing Strategy
1. Test Step 1 training with dummy data
2. Test Step 2 loading Step 1 checkpoint
3. Verify freeze/train parameter counts
4. Check forward pass outputs

### Dataset Configuration
- Step 1: TartanAir, MVS-Synth, Spring, etc. (518×518)
- Step 2: Same or subset (518×518 for now, hybrid 2K training later)
