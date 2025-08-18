#!/usr/bin/env python3

import pyrootutils
root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))

import os
import sys
import torch
import numpy as np
import laspy
from laspy import ExtraBytesParams
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.models.semantic import SemanticSegmentationModule

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations:
# https://omegaconf.readthedocs.io/en/2.1_branch/how_to_guides.html#
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

def main():
    # Checkpoint path
    checkpoint_path = "/home/moussabendjilali/spt-2_dales.ckpt"
    input_las = "/home/moussabendjilali/visualization.las"
    output_las = "/home/moussabendjilali/visualization_spt_segmented.las"

    # Compose config to get transforms only - don't prepare dataset
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["datamodule=semantic/dales.yaml"])
    
    # Instantiate model from config first, then load checkpoint
    model = instantiate(cfg.model)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read input LAS file directly
    las = laspy.read(input_las)
    gt = np.array(las.sem_class).astype(np.int16)

    # Create Data object directly from LAS
    from src.data import Data
    from src.data.nag import NAGBatch, NAG
    from src.transforms import instantiate_transforms

    data = Data()
    # Positions (subtract offset for numerical stability)
    pos = torch.from_numpy(np.vstack([las.x, las.y, las.z]).T).float()
    pos_offset = pos[0]
    data.pos = pos - pos_offset
    data.pos_offset = pos_offset


    data.intensity = torch.from_numpy(np.array(las.intensity).astype(float))
    # Apply transforms from config
    pre_transforms = instantiate_transforms(cfg.datamodule.pre_transform)
    on_device_test_transforms = instantiate_transforms(cfg.datamodule.on_device_test_transform)
    
    # Apply preprocessing transforms
    if pre_transforms:
        data = pre_transforms(data)
    
    # Convert to NAG if needed
    nag = data if isinstance(data, NAG) else NAG([data])
    
    # Create batch and move to device
    batch = NAGBatch.from_nag_list([nag])
    batch = batch.to(device)
    
    # Apply on-device transforms
    if on_device_test_transforms:
        batch = on_device_test_transforms(batch)

    # Run inference
    with torch.no_grad():
        output = model(batch)

    # Extract predictions
    logits = output.logits[0] if output.multi_stage else output.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int16) + 1

    # Map predictions back to full resolution points
    if hasattr(batch[0], 'super_index'):
        super_index = batch[0].super_index.cpu().numpy()
        pred_full = preds[super_index]
    else:
        # If no hierarchical structure, predictions are already at point level
        pred_full = preds

    # Ensure pred_full has same length as ground truth
    if len(pred_full) != len(gt):
        print(f"Warning: prediction length {len(pred_full)} != ground truth length {len(gt)}")
        # Pad predictions to match GT length (repeat last prediction)
        if len(pred_full) < len(gt):
            # Pad with last prediction value
            last_pred = pred_full[-1] if len(pred_full) > 0 else 0
            padding = np.full(len(gt) - len(pred_full), last_pred, dtype=np.int16)
            pred_full = np.concatenate([pred_full, padding])
        else:
            # Truncate to GT length
            pred_full = pred_full[:len(gt)]

    # Compute error flag
    error = (np.array(pred_full) != np.array(gt)).astype(np.int16)
    
    # Create a copy of the original header
    new_header = laspy.LasHeader(version="1.4")
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales
    new_header.point_format = las.header.point_format
    
    # Add extra byte dimensions to header
    extra_dims = [
        ExtraBytesParams(name="pred", type="int16"), 
        ExtraBytesParams(name="error", type="int16")
    ]
    
    # Add extra byte dimensions to header, skipping duplicates
    for dim in extra_dims:
        try:
            new_header.add_extra_dim(dim)
        except Exception:
            print(f"Skipping duplicate extra dim '{dim.name}'")
    new_header.add_extra_dim(ExtraBytesParams(name="cluster", type="int16"))
    # Create new LAS data object with the modified header
    new_las = laspy.LasData(new_header)
    
    # Copy all original point data
    new_las.x = las.x
    new_las.y = las.y  
    new_las.z = las.z
    if hasattr(las, 'intensity'):
        new_las.intensity = las.intensity
    if hasattr(las, 'classification'):
        new_las.classification = las.classification
    if hasattr(las, 'return_number'):
        new_las.return_number = las.return_number
    if hasattr(las, 'number_of_returns'):
        new_las.number_of_returns = las.number_of_returns
    
    # Add new extra byte fields
    new_las.sem_class = gt
    new_las.pred = pred_full  
    new_las.error = error
    new_las.cluster = batch[2].super_index[batch[1].super_index[batch[0].super_index]].cpu().numpy()
    
    # Write new LAS file
    new_las.write(output_las)

    print(f"Saved segmented LAS to {output_las}")


if __name__ == "__main__":
    main() 