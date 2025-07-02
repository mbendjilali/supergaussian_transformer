# hGMM Visualization

This directory contains scripts for visualizing the results of hierarchical Gaussian Mixture Model (hGMM) clustering with the GEM variant.

## Files

- `test_hgmm_visualization.py` - Main visualization script
- `example_visualization.py` - Example usage script
- `README_visualization.md` - This file

## Requirements

The visualization requires the following Python packages:
- `plotly` - For interactive 3D visualization
- `laspy` - For reading LAS point cloud files
- `torch` - For tensor operations
- `numpy` - For numerical operations

Install with:
```bash
pip install plotly laspy torch numpy
```

## Usage

### Basic Usage

```bash
python test_hgmm_visualization.py --input path/to/pointcloud.las
```

### Advanced Usage

```bash
python test_hgmm_visualization.py \
    --input data/pointcloud.las \
    --output visualization.html \
    --cluster_size 128 \
    --depth 4 \
    --max_iter 5 \
    --alpha 1.0 \
    --tol 0.01 \
    --downsample 0.5
```

### Using the Example Script

```bash
python example_visualization.py data/pointcloud.las
```

## Parameters

- `--input`: Input LAS file path (required)
- `--output`: Output HTML file path (optional, auto-generated if not provided)
- `--cluster_size`: Target cluster size (default: 192)
- `--depth`: Hierarchy depth (default: 4)
- `--alpha`: Regularization parameter (default: 1.0)
- `--tol`: Convergence tolerance (default: 0.01)
- `--max_iter`: Maximum EM iterations (default: 5)
- `--downsample`: Grid size for downsampling (optional)
- `--sample_rate`: Sample every Nth point for visualization (default: 20, higher = smaller file)

## File Size Optimization

The visualization is optimized for small file sizes:

- **Point Sampling**: Only displays every Nth point (controlled by `--sample_rate`)
- **Gaussian Ellipsoids**: Limited to 3 per level with minimal point count
- **CDN Loading**: Uses Plotly CDN instead of embedding the full library
- **Minimal Hover Info**: Disabled hover tooltips to reduce data
- **Reduced Colors**: Uses only 5 basic colors instead of complex palettes

For very large point clouds, use:
```bash
python test_hgmm_visualization.py --input large_cloud.las --sample_rate 100 --downsample 2.0
```

## Output

The script generates an interactive HTML file that shows:

1. **Point Cloud Visualization**: Points colored by cluster assignment
2. **Gaussian Ellipsoids**: 3D ellipsoids representing the fitted Gaussians
3. **Multiple Levels**: Each subplot shows a different hierarchy level
4. **Interactive Features**: 
   - Rotate, zoom, and pan the 3D view
   - Hover over points to see cluster information
   - Hover over ellipsoids to see Gaussian information

## Visualization Features

- **Cluster Coloring**: Each cluster is assigned a unique color from a qualitative color palette
- **Gaussian Ellipsoids**: Semi-transparent ellipsoids show the shape and orientation of fitted Gaussians
- **Level Comparison**: Side-by-side comparison of different hierarchy levels
- **Interactive Controls**: Full 3D navigation and hover information

## Example Output

The visualization will show:
- Level 0: Coarse clustering with few large clusters
- Level 1: Medium-level clustering
- Level 2: Fine clustering with many small clusters
- Level 3: Finest level clustering

Each level displays both the point cloud (colored by cluster) and the corresponding Gaussian ellipsoids.

## Tips

1. **For Large Point Clouds**: Use the `--downsample` parameter to reduce the number of points for faster processing
2. **For Better Visualization**: Use smaller `--cluster_size` values for clearer cluster boundaries
3. **For Faster Processing**: Reduce `--max_iter` and `--depth` parameters
4. **Browser Compatibility**: The HTML file works best in modern browsers (Chrome, Firefox, Safari, Edge)

## Troubleshooting

- **Memory Issues**: Reduce point cloud size using downsampling
- **Slow Processing**: Reduce cluster size and number of iterations
- **Visualization Issues**: Ensure you have a modern web browser with WebGL support 