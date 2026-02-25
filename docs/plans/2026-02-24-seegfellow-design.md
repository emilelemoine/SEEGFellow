# SEEGFellow — Design Document

## Overview

A 3D Slicer scripted module for semi-automatic localization and visualization of SEEG (stereo-electroencephalography) electrode contacts from post-implant CT images.

## Input Assumptions

- A post-implant CT already coregistered to a T1 MRI (registration handled outside this extension)
- The CT volume is loaded in Slicer as a scalar volume node

## User Workflow

1. Select the CT volume from a dropdown
2. Create a new electrode: set name (e.g., "A", "LT"), number of contacts k, and electrode parameters (contact length, spacing, diameter) from presets or custom values
3. Click "Place seed point" → Slicer enters markup placement mode. User clicks on the **deepest (most mesial) contact** (#1) in the slice view
4. Optionally click "Set direction hint" → user places a second point along the electrode shaft to help the algorithm find the correct trajectory direction
5. Click "Detect contacts" → algorithm runs, places k labeled fiducials (e.g., A1, A2, ..., Ak)
6. User can **drag any fiducial** to manually correct positions
7. Optionally create a segmentation with per-contact cylindrical segments
8. Repeat for each electrode

## Extension Structure

```
SEEGFellow/
├── CMakeLists.txt
├── SEEGFellow/
│   ├── CMakeLists.txt
│   ├── SEEGFellow.py                  # Module, Widget, Test classes
│   ├── SEEGFellowLib/
│   │   ├── __init__.py
│   │   ├── electrode_model.py         # Data classes: Electrode, Contact, ElectrodeParams
│   │   ├── trajectory_detector.py     # Base class + IntensityProfileDetector
│   │   └── contact_segmenter.py       # Creates segmentation from detected contacts
│   ├── Resources/
│   │   ├── Icons/
│   │   │   └── SEEGFellow.png
│   │   └── UI/
│   │       └── SEEGFellow.ui
│   └── Testing/
│       └── Python/
│           └── SEEGFellowTest.py
└── README.md
```

## Data Model (`electrode_model.py`)

```python
@dataclass
class Contact:
    index: int           # 1-based (1 = deepest/mesial)
    position_ras: tuple  # (R, A, S) coordinates in mm
    label: str           # e.g., "A1", "LT3"

@dataclass
class ElectrodeParams:
    contact_length: float    # mm (e.g., 2.0)
    contact_spacing: float   # mm center-to-center (e.g., 3.5)
    contact_diameter: float  # mm (e.g., 0.8)

@dataclass
class Electrode:
    name: str                    # e.g., "A", "LT", "SFG"
    num_contacts: int            # k
    params: ElectrodeParams
    contacts: list[Contact]
    trajectory_direction: tuple  # unit vector from mesial to lateral
    seed_point: tuple            # original user click (RAS)
    markups_node_id: str         # Slicer MRML node ID for the fiducials
```

## Detection Algorithm (`trajectory_detector.py`)

### Base class

```python
class TrajectoryDetector(ABC):
    @abstractmethod
    def detect(self, seed_ras, volume_node, num_contacts, params, direction_hint=None) -> list[Contact]:
        """Given a seed point, find num_contacts contacts along an electrode."""
```

This abstraction allows swapping in a future DL-based detector without changing the rest of the module.

### IntensityProfileDetector (initial implementation)

1. **Seed point**: User-clicked RAS coordinates of contact #1
2. **Local neighborhood**: Extract voxels within a ~20mm radius sphere around the seed. Threshold at a configurable HU value (default ~2500) to isolate metal
3. **Trajectory estimation**:
   - If direction hint provided: use the vector from seed to hint point
   - Otherwise: run PCA on the thresholded voxel positions. The first principal component gives the electrode axis direction
   - Orient the direction vector to point outward (away from brain center / toward lateral)
4. **Intensity profile**: Sample CT intensity along the trajectory line from contact #1, extending outward for `(k-1) * spacing + margin` mm. Use sub-voxel interpolation for accuracy
5. **Peak detection**: Find peaks in the intensity profile using scipy.signal.find_peaks with `distance` based on expected contact spacing
6. **Refinement**: If detected peaks are within a tolerance of the expected evenly-spaced positions, snap to the expected grid. This handles noisy profiles where some peaks are ambiguous
7. **Output**: k Contact objects with RAS positions, numbered 1 (deepest) to k (most lateral)

### Key parameters (configurable in GUI)

- `intensity_threshold`: HU threshold for metal detection (default: 2500)
- `search_radius`: radius in mm for local neighborhood PCA (default: 20)
- `peak_prominence`: minimum prominence for peak detection (default: 500)
- `spacing_tolerance`: max deviation from expected spacing before rejecting a peak (default: 1.0 mm)

## GUI Design (`SEEGFellow.ui` + Widget)

### Panel sections (top to bottom)

1. **Input**: `qMRMLNodeComboBox` for CT volume selection (filtered to scalar volumes)
2. **Electrode Parameters**:
   - Name: `QLineEdit`
   - Number of contacts: `QSpinBox` (range 1–20)
   - Preset dropdown: `QComboBox` with entries like "AdTech SD08R-SP10X (2mm contact, 3.5mm spacing)" etc.
   - Contact length / spacing / diameter: `QDoubleSpinBox` fields (auto-filled from preset, editable for custom)
3. **Detection**:
   - "Place seed point" `QPushButton` → enters single-fiducial placement mode
   - "Set direction hint" `QPushButton` → enters second-point placement mode (optional)
   - Intensity threshold: `ctkSliderWidget` (range 500–4000, default 2500)
   - "Detect contacts" `QPushButton` → runs algorithm
4. **Results**:
   - Contact table: `QTableWidget` showing contact label, R, A, S coordinates
   - "Create segmentation" `QPushButton`
5. **Electrode List**:
   - `QListWidget` showing all electrodes in the session
   - "Delete electrode" button to remove selected electrode and its fiducials

## Segmentation (`contact_segmenter.py`)

Creates a `vtkMRMLSegmentationNode` with one segment per contact:

- For each contact, generate a small cylinder (using contact_diameter and contact_length along the trajectory direction)
- Mark voxels inside the cylinder in a binary labelmap
- Each segment named to match the contact label (e.g., "A1")
- Distinct colors per electrode (all contacts of electrode "A" share one color, "B" another, etc.)
- Generates a closed surface representation for 3D visualization

## Slicer Integration Details

- **Fiducial interaction**: Each electrode gets its own `vtkMRMLMarkupsFiducialNode`. Control points are draggable by default, giving free manual correction
- **Observation**: The widget observes `PointPositionDefinedEvent` on the seed markup to trigger detection when the user places a point
- **Module loading**: During development, add the `SEEGFellow/` directory to Slicer's additional module paths. No compilation needed for pure-Python scripted modules
- **Dependencies**: numpy (bundled with Slicer), scipy (may need `pip_install('scipy')` via Slicer's Python). vtk and slicer are available in the Slicer environment

## Future Extensibility

The modular design supports:

- **DL-based auto-detection**: Implement a new `TrajectoryDetector` subclass that takes an electrode table (entry/target pairs) and a CT volume, runs a neural network, and returns all contacts for all electrodes at once
- **CSV/spreadsheet export**: Add an export method to `Electrode` that writes contact labels + RAS coordinates
- **Atlas mapping**: Post-processing step to map contact positions to MNI space or brain atlas regions
- **Batch processing**: The Logic class is GUI-independent, so it can be called from Slicer's Python console or scripted pipelines
