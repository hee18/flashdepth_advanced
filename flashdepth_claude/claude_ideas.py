class SpatialGSP(nn.Module):
      """Spatial-varying Global Scale Predictor"""
      def __init__(self, dim=1024, num_classes=19):  # 19 = Cityscapes classes
          super().__init__()

          # 1. Semantic segmentation branch (lightweight)
          self.seg_head = nn.Sequential(
              nn.Conv2d(dim, 256, 1),
              nn.ReLU(),
              nn.Conv2d(256, num_classes, 1)
          )

          # 2. Class-specific scale/shift predictor
          self.class_gsp = nn.ModuleList([
              nn.Sequential(
                  nn.Linear(dim, 128),
                  nn.ReLU(),
                  nn.Linear(128, 2)  # [scale, shift] per class
              ) for _ in range(num_classes)
          ])

          # 3. Global GSP (fallback)
          self.global_gsp = nn.Sequential(
              nn.Linear(dim, 256),
              nn.ReLU(),
              nn.Linear(256, 2)
          )

      def forward(self, cls_token, patch_tokens, relative_depth):
          """
          Args:
              cls_token: (B, 1024) - global scene understanding
              patch_tokens: (B, N, 1024) - spatial features
              relative_depth: (B, H, W) - relative depth map
          Returns:
              metric_depth: (B, H, W) - spatially-varying corrected depth
          """
          B, N, C = patch_tokens.shape
          H_p, W_p = int(np.sqrt(N)), int(np.sqrt(N))

          # 1. Predict semantic segmentation (lightweight, no external supervision needed initially)
          patch_feat = rearrange(patch_tokens, 'b (h w) c -> b c h w', h=H_p, w=W_p)
          seg_logits = self.seg_head(patch_feat)  # (B, num_classes, H_p, W_p)
          seg_probs = F.softmax(seg_logits, dim=1)

          # Upsample to match depth resolution
          seg_probs = F.interpolate(seg_probs, size=relative_depth.shape[-2:],
                                    mode='bilinear', align_corners=True)

          # 2. Predict class-specific scale/shift
          class_params = []
          for i, gsp_module in enumerate(self.class_gsp):
              # Use class-weighted average of patch tokens
              class_weight = seg_probs[:, i:i+1, :, :]  # (B, 1, H, W)

              # Pool features with semantic weights
              weighted_feat = (patch_feat * class_weight).sum(dim=(2,3)) / (class_weight.sum(dim=(2,3)) + 1e-6)
              params = gsp_module(weighted_feat)  # (B, 2)
              class_params.append(params)

          class_params = torch.stack(class_params, dim=1)  # (B, num_classes, 2)

          # 3. Global fallback
          global_params = self.global_gsp(cls_token)  # (B, 2)

          # 4. Blend: class-specific + global
          scale_map = torch.zeros_like(relative_depth)
          shift_map = torch.zeros_like(relative_depth)

          for i in range(num_classes):
              class_scale = F.softplus(class_params[:, i, 0])  # (B,)
              class_shift = class_params[:, i, 1]  # (B,)

              scale_map += seg_probs[:, i] * class_scale[:, None, None]
              shift_map += seg_probs[:, i] * class_shift[:, None, None]

          # Add global component
          global_scale = F.softplus(global_params[:, 0])
          global_shift = global_params[:, 1]

          # Weighted blend (learnable)
          metric_depth = (scale_map + 0.1 * global_scale[:, None, None]) * relative_depth + \
                         (shift_map + 0.1 * global_shift[:, None, None])

          return metric_depth, seg_probs
      
      
class AttentionGSP(nn.Module):
      """Attention-based spatially-adaptive GSP"""
      def __init__(self, dim=1024, num_heads=8):
          super().__init__()

          # Multi-head attention: CLS token queries patch tokens
          self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

          # Spatial importance predictor
          self.importance_head = nn.Sequential(
              nn.Linear(dim, 256),
              nn.ReLU(),
              nn.Linear(256, 1),
              nn.Sigmoid()
          )

          # Region-specific GSP
          self.foreground_gsp = nn.Sequential(
              nn.Linear(dim, 128),
              nn.ReLU(),
              nn.Linear(128, 2)
          )

          self.background_gsp = nn.Sequential(
              nn.Linear(dim, 128),
              nn.ReLU(),
              nn.Linear(128, 2)
          )

      def forward(self, cls_token, patch_tokens, relative_depth):
          B, N, C = patch_tokens.shape
          H_p, W_p = int(np.sqrt(N)), int(np.sqrt(N))

          # 1. Compute attention: which patches does CLS token attend to?
          attn_output, attn_weights = self.attention(
              cls_token.unsqueeze(1),  # Query: (B, 1, C)
              patch_tokens,             # Key/Value: (B, N, C)
              patch_tokens
          )
          attn_weights = attn_weights.squeeze(1)  # (B, N)

          # 2. Predict spatial importance (foreground vs background)
          importance = self.importance_head(patch_tokens)  # (B, N, 1)
          importance_map = rearrange(importance, 'b (h w) c -> b c h w', h=H_p, w=W_p)
          importance_map = F.interpolate(importance_map, size=relative_depth.shape[-2:],
                                        mode='bilinear', align_corners=True).squeeze(1)

          # 3. Foreground/background features
          fg_mask = importance_map > 0.5
          bg_mask = ~fg_mask

          # Weighted pooling
          fg_feat = (patch_tokens * importance).sum(dim=1) / (importance.sum(dim=1) + 1e-6)
          bg_feat = (patch_tokens * (1 - importance)).sum(dim=1) / ((1 - importance).sum(dim=1) + 1e-6)

          # 4. Predict scale/shift
          fg_params = self.foreground_gsp(fg_feat)  # (B, 2)
          bg_params = self.background_gsp(bg_feat)  # (B, 2)

          fg_scale, fg_shift = F.softplus(fg_params[:, 0]), fg_params[:, 1]
          bg_scale, bg_shift = F.softplus(bg_params[:, 0]), bg_params[:, 1]

          # 5. Apply spatially-varying transformation
          scale_map = fg_scale[:, None, None] * importance_map + \
                      bg_scale[:, None, None] * (1 - importance_map)
          shift_map = fg_shift[:, None, None] * importance_map + \
                      bg_shift[:, None, None] * (1 - importance_map)

          metric_depth = scale_map * relative_depth + shift_map

          return metric_depth, importance_map
    

class TemporalAwareGSP(nn.Module):
      """Use Mamba temporal states for consistency"""
      def __init__(self, dim=1024, mamba_dim=256):
          super().__init__()

          # Temporal feature fusion
          self.temporal_fusion = nn.Sequential(
              nn.Linear(dim + mamba_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 2)
          )

          # Spatial GSP
          self.spatial_gsp = nn.Conv2d(dim, 2, kernel_size=1)

      def forward(self, cls_token, patch_tokens, mamba_states, relative_depth):
          """
          Args:
              mamba_states: (B, T, mamba_dim) - temporal features from Mamba
          """
          # 1. Fuse CLS token with temporal context
          temporal_context = mamba_states.mean(dim=1)  # (B, mamba_dim)
          fused_feat = torch.cat([cls_token, temporal_context], dim=-1)

          # 2. Temporal-aware global params
          global_params = self.temporal_fusion(fused_feat)

          # 3. Spatial-varying params
          B, N, C = patch_tokens.shape
          H_p, W_p = int(np.sqrt(N)), int(np.sqrt(N))
          patch_feat = rearrange(patch_tokens, 'b (h w) c -> b c h w', h=H_p, w=W_p)
          spatial_params = self.spatial_gsp(patch_feat)  # (B, 2, H_p, W_p)
          spatial_params = F.interpolate(spatial_params, size=relative_depth.shape[-2:],
                                         mode='bilinear', align_corners=True)

          # 4. Combine
          scale = F.softplus(global_params[:, 0:1, None, None] + spatial_params[:, 0:1])
          shift = global_params[:, 1:2, None, None] + spatial_params[:, 1:2]

          metric_depth = scale * relative_depth + shift
          return metric_depth
