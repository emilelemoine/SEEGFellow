# SEEGFellow

A 3D Slicer scripted module for semi-automatic SEEG electrode localization from post-implant CT.

## Features

- Metal threshold segmentation to isolate electrode contacts from CT
- Brain mask generation via SynthSeg segmentation
- Automated contact detection using Laplacian of Gaussian blob detection
- Electrode grouping via RANSAC line fitting
- Contact anatomical labeling using DKT atlas parcellation
- CSV export of contact positions and labels

## Requirements

- 3D Slicer with SlicerFreeSurfer extension (for SynthSeg)
- A pre-implant T1 MRI registered to post-implant CT (or vice versa)

## Usage

1. Load the post-implant CT and pre-implant T1 in Slicer
2. Register the CT to the T1 (e.g., using the General Registration module)
3. Open the SEEGFellow module
4. Set the CT threshold and run metal segmentation
5. Run brain segmentation (SynthSeg)
6. Run electrode detection
7. (Optional) Run contact labeling
8. Export results as CSV

## Development

```bash
uv sync
source .venv/bin/activate
pytest tests/ -v
```
