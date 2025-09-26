
import torch
ckpt = torch.load('train_results/results_4/best_metric_head.pth', map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
print('Model state dict keys (first 20):')
model_keys = list(ckpt['model_state_dict'].keys())[:20]
for k in model_keys: print(f'  {k}')
print('GSP head keys:')
gsp_keys = [k for k in ckpt['model_state_dict'].keys() if 'gsp_head' in k]
for k in gsp_keys: print(f'  {k}')
if not gsp_keys: print('  No GSP head keys found!')

