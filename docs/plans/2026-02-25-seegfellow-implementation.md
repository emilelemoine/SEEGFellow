# SEEGFellow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 3D Slicer scripted module that guides users through semi-automatic SEEG electrode localization from post-implant CT.

**Architecture:** A wizard-style Slicer module with collapsible step panels. Algorithm code operates on numpy arrays for testability, with thin Slicer wrappers handling node ↔ array conversion. Each pipeline step (registration, segmentation, detection) is a separate library module.

**Tech Stack:** Python 3 (Slicer's bundled interpreter), numpy (bundled), scipy (pip_install at runtime), vtk + slicer APIs, BRAINSFit CLI module (bundled with Slicer).

**Design doc:** `docs/plans/2026-02-24-seegfellow-design.md`

---

## Branch 1: `feat/scaffolding-and-data-model`

Sets up the extension directory structure, build files, dev tooling, and the core data model.

### Task 1.1: Create extension directory structure

**Files:**
- Create: `SEEGFellow/CMakeLists.txt`
- Create: `SEEGFellow/SEEGFellow/CMakeLists.txt`
- Create: `SEEGFellow/SEEGFellow/SEEGFellow.py`
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py`
- Create: `SEEGFellow/SEEGFellow/Resources/Icons/SEEGFellow.png`
- Create: `SEEGFellow/SEEGFellow/Testing/Python/SEEGFellowTest.py`

**Step 1: Create the top-level CMakeLists.txt**

```cmake
# SEEGFellow/CMakeLists.txt
cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)
project(SEEGFellow)

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(SEEGFellow)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
```

**Step 2: Create the module CMakeLists.txt**

```cmake
# SEEGFellow/SEEGFellow/CMakeLists.txt
set(MODULE_NAME SEEGFellow)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/electrode_model.py
  ${MODULE_NAME}Lib/registration.py
  ${MODULE_NAME}Lib/metal_segmenter.py
  ${MODULE_NAME}Lib/electrode_detector.py
  ${MODULE_NAME}Lib/trajectory_detector.py
  ${MODULE_NAME}Lib/contact_segmenter.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
```

**Step 3: Create module skeleton `SEEGFellow.py`**

```python
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


class SEEGFellow(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SEEGFellow"
        self.parent.categories = ["Neuro"]
        self.parent.contributors = [""]
        self.parent.helpText = (
            "Semi-automatic SEEG electrode localization from post-implant CT."
        )
        self.parent.acknowledgementText = ""


class SEEGFellowWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # TODO: build wizard UI in later branch

    def cleanup(self):
        pass


class SEEGFellowLogic(ScriptedLoadableModuleLogic):
    pass


class SEEGFellowTest(ScriptedLoadableModuleTest):
    def runTest(self):
        pass
```

**Step 4: Create empty `SEEGFellowLib/__init__.py`**

```python
# SEEGFellow library modules
```

**Step 5: Create placeholder icon**

Create a 128x128 PNG at `Resources/Icons/SEEGFellow.png`. Use Slicer's default module icon placeholder or a simple solid-color square.

**Step 6: Create test skeleton**

```python
# Testing/Python/SEEGFellowTest.py
# Slicer integration tests — run inside Slicer's test runner
```

**Step 7: Commit**

```bash
git add SEEGFellow/
git commit -m "feat: create extension scaffolding"
```

---

### Task 1.2: Set up dev tooling (pyproject.toml, pytest)

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

This is for dev/test tooling only — Slicer loads the module directly, not via pip.

```toml
[project]
name = "seegfellow-dev"
version = "0.0.0"
requires-python = ">=3.9"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.pyright]
venvPath = "."
venv = ".venv"
```

**Step 2: Create .envrc for direnv**

```bash
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
```

**Step 3: Set up venv and install dev deps**

```bash
uv venv
direnv allow
uv pip install pytest numpy scipy
```

**Step 4: Create empty `tests/__init__.py`**

**Step 5: Verify pytest runs**

Run: `pytest --co`
Expected: "no tests ran" (no test files yet)

**Step 6: Commit**

```bash
git add pyproject.toml .envrc tests/
git commit -m "chore: add dev tooling (pytest, pyright, direnv)"
```

---

### Task 1.3: Implement data model with tests

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py`
- Create: `tests/test_electrode_model.py`

**Step 1: Write tests for data model**

```python
# tests/test_electrode_model.py
import sys
import os

# Add the module source to the path so we can import without Slicer
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams


class TestContact:
    def test_create_contact(self):
        c = Contact(index=1, position_ras=(10.0, 20.0, 30.0), label="A1")
        assert c.index == 1
        assert c.position_ras == (10.0, 20.0, 30.0)
        assert c.label == "A1"

    def test_default_label_is_empty(self):
        c = Contact(index=3, position_ras=(0.0, 0.0, 0.0))
        assert c.label == ""


class TestElectrodeParams:
    def test_params_without_gaps(self):
        p = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8
        )
        assert p.gap_spacing is None
        assert p.contacts_per_group is None

    def test_params_with_gaps(self):
        p = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=3.5,
            contact_diameter=0.8,
            gap_spacing=10.0,
            contacts_per_group=6,
        )
        assert p.gap_spacing == 10.0
        assert p.contacts_per_group == 6


class TestElectrode:
    def test_create_electrode(self):
        params = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8
        )
        contacts = [
            Contact(index=i, position_ras=(float(i), 0.0, 0.0))
            for i in range(1, 7)
        ]
        e = Electrode(
            name="A",
            params=params,
            contacts=contacts,
            trajectory_direction=(1.0, 0.0, 0.0),
        )
        assert e.name == "A"
        assert e.num_contacts == 6
        assert len(e.contacts) == 6

    def test_assign_labels(self):
        params = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8
        )
        contacts = [
            Contact(index=i, position_ras=(float(i), 0.0, 0.0))
            for i in range(1, 4)
        ]
        e = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=(1.0, 0.0, 0.0),
        )
        e.assign_labels("LT")
        assert e.name == "LT"
        assert e.contacts[0].label == "LT1"
        assert e.contacts[2].label == "LT3"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_electrode_model.py -v`
Expected: FAIL — `electrode_model` module doesn't exist yet

**Step 3: Implement the data model**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py
from dataclasses import dataclass, field


@dataclass
class Contact:
    """A single electrode contact.

    Example::

        c = Contact(index=1, position_ras=(10.0, 20.0, 30.0), label="A1")
    """

    index: int  # 1-based (1 = deepest/mesial)
    position_ras: tuple[float, float, float]  # (R, A, S) in mm
    label: str = ""


@dataclass
class ElectrodeParams:
    """Physical parameters of an electrode model.

    Example::

        # Standard electrode without gaps
        p = ElectrodeParams(contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8)

        # Electrode with gap between groups of 6 contacts
        p = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8,
            gap_spacing=10.0, contacts_per_group=6,
        )
    """

    contact_length: float  # mm
    contact_spacing: float  # mm center-to-center
    contact_diameter: float  # mm
    gap_spacing: float | None = None  # mm center-to-center between groups
    contacts_per_group: int | None = None  # contacts per group


@dataclass
class Electrode:
    """A detected electrode with its contacts.

    Example::

        e = Electrode(
            name="A",
            params=ElectrodeParams(contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8),
            contacts=[Contact(index=1, position_ras=(10.0, 20.0, 30.0))],
            trajectory_direction=(1.0, 0.0, 0.0),
        )
        e.assign_labels("A")  # contacts become A1, A2, ...
    """

    name: str
    params: ElectrodeParams
    contacts: list[Contact] = field(default_factory=list)
    trajectory_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    markups_node_id: str = ""

    @property
    def num_contacts(self) -> int:
        return len(self.contacts)

    def assign_labels(self, name: str) -> None:
        """Set electrode name and update all contact labels."""
        self.name = name
        for contact in self.contacts:
            contact.label = f"{name}{contact.index}"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_electrode_model.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py tests/test_electrode_model.py
git commit -m "feat: add electrode data model (Contact, ElectrodeParams, Electrode)"
```

**Step 6: Merge branch**

```bash
git checkout main
git merge feat/scaffolding-and-data-model
git branch -d feat/scaffolding-and-data-model
```

---

## Branch 2: `feat/metal-segmentation`

Implements the metal segmentation algorithm (Step 4 in the design).

### Task 2.1: Core metal segmentation algorithm (numpy)

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py`
- Create: `tests/test_metal_segmenter.py`

**Step 1: Write tests for the core thresholding + cleanup logic**

The core functions operate on numpy arrays so they're testable outside Slicer. We create synthetic CT-like arrays with known metal blobs and bone-like structures.

```python
# tests/test_metal_segmenter.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.metal_segmenter import threshold_volume, cleanup_metal_mask


class TestThresholdVolume:
    def test_basic_threshold(self):
        vol = np.zeros((20, 20, 20), dtype=np.float32)
        vol[10, 10, 10] = 3000.0  # metal voxel
        vol[5, 5, 5] = 1000.0  # bone voxel (below threshold)
        mask = threshold_volume(vol, threshold=2500)
        assert mask[10, 10, 10] == 1
        assert mask[5, 5, 5] == 0

    def test_threshold_configurable(self):
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[5, 5, 5] = 2000.0
        assert threshold_volume(vol, threshold=1500)[5, 5, 5] == 1
        assert threshold_volume(vol, threshold=2500)[5, 5, 5] == 0


class TestCleanupMetalMask:
    def test_removes_small_noise(self):
        """Single isolated voxel should be removed by morphological opening."""
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[15, 15, 15] = 1  # single noise voxel
        # Add a real cluster (line of contacts)
        for i in range(10, 20):
            mask[i, 15, 15] = 1
            mask[i, 16, 15] = 1
        cleaned = cleanup_metal_mask(mask)
        # The real cluster should survive
        assert cleaned[14, 15, 15] == 1 or cleaned[15, 15, 15] == 1
        # The isolated noise voxel far from the cluster should be gone
        # (it's actually within the cluster here, so let's use a truly isolated one)
        mask2 = np.zeros((30, 30, 30), dtype=np.uint8)
        mask2[2, 2, 2] = 1  # isolated noise
        for i in range(15, 25):
            mask2[i, 15, 15] = 1
            mask2[i, 16, 15] = 1
        cleaned2 = cleanup_metal_mask(mask2)
        assert cleaned2[2, 2, 2] == 0

    def test_removes_large_bone_blobs(self):
        """Large bulky components (bone) should be removed."""
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        # Electrode-like: thin elongated cluster
        for i in range(10, 30):
            mask[i, 25, 25] = 1
            mask[i, 26, 25] = 1
        # Bone-like: large bulky blob
        mask[35:45, 35:45, 35:45] = 1
        cleaned = cleanup_metal_mask(mask)
        # Electrode should survive
        assert np.any(cleaned[10:30, 25:27, 25] == 1)
        # Bone blob should be removed
        assert np.sum(cleaned[35:45, 35:45, 35:45]) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metal_segmenter.py -v`
Expected: FAIL — module not found

**Step 3: Implement core functions**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py
"""Metal segmentation from CT volume.

Core algorithm functions operate on numpy arrays for testability.
The MetalSegmenter class wraps these for use with Slicer volume nodes.
"""

import numpy as np
from scipy import ndimage


def threshold_volume(volume: np.ndarray, threshold: float = 2500) -> np.ndarray:
    """Threshold a CT volume to isolate high-intensity voxels.

    Example::

        mask = threshold_volume(ct_array, threshold=2500)
    """
    return (volume >= threshold).astype(np.uint8)


def cleanup_metal_mask(
    mask: np.ndarray,
    min_component_size: int = 5,
    max_component_volume: int = 5000,
    max_elongation_ratio: float = 2.0,
) -> np.ndarray:
    """Remove noise and bone fragments from a binary metal mask.

    Steps:
    1. Morphological opening to remove small noise
    2. Connected component analysis
    3. Remove components smaller than min_component_size
    4. Remove large bulky components (bone) based on volume and shape

    Args:
        mask: Binary mask (0/1 uint8).
        min_component_size: Minimum voxel count to keep a component.
        max_component_volume: Components larger than this are candidate bone.
        max_elongation_ratio: Bone has low elongation (bounding box is roughly
            cubic). Components with (longest axis / shortest axis) below this
            ratio AND volume above max_component_volume are removed.

    Example::

        cleaned = cleanup_metal_mask(binary_mask, min_component_size=5)
    """
    struct = ndimage.generate_binary_structure(3, 1)

    # Morphological opening: removes small noise
    opened = ndimage.binary_opening(mask, structure=struct, iterations=1).astype(
        np.uint8
    )

    # Morphological closing: fills small holes within contacts
    closed = ndimage.binary_closing(opened, structure=struct, iterations=1).astype(
        np.uint8
    )

    # Connected component analysis
    labeled, num_features = ndimage.label(closed, structure=struct)

    result = np.zeros_like(mask)
    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id
        volume = np.sum(component)

        # Remove small noise
        if volume < min_component_size:
            continue

        # Check if large component is bone-like (bulky, not elongated)
        if volume > max_component_volume:
            coords = np.argwhere(component)
            bbox_size = coords.max(axis=0) - coords.min(axis=0) + 1
            sorted_dims = np.sort(bbox_size)
            # Elongation: longest / shortest dimension
            if sorted_dims[0] > 0:
                elongation = sorted_dims[-1] / sorted_dims[0]
            else:
                elongation = 1.0
            # Bone is bulky (low elongation), electrodes are elongated
            if elongation < max_elongation_ratio:
                continue

        result[component] = 1

    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metal_segmenter.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py tests/test_metal_segmenter.py
git commit -m "feat: add metal segmentation core algorithm"
```

---

### Task 2.2: Slicer wrapper for metal segmentation

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py` (append class)

**Step 1: Add MetalSegmenter class**

Append to `metal_segmenter.py`:

```python
class MetalSegmenter:
    """Slicer wrapper: segments metal from a CT volume node.

    Example (in Slicer Python console)::

        segmenter = MetalSegmenter()
        metal_labelmap = segmenter.segment(ct_volume_node, threshold=2500)
    """

    def segment(self, ct_volume_node, threshold: float = 2500):
        """Threshold CT and clean up to isolate metal voxels.

        Args:
            ct_volume_node: vtkMRMLScalarVolumeNode with the CT data.
            threshold: HU threshold for metal.

        Returns:
            vtkMRMLLabelMapVolumeNode containing the metal mask.
        """
        import slicer
        from slicer.util import arrayFromVolume, updateVolumeFromArray

        ct_array = arrayFromVolume(ct_volume_node)
        mask = threshold_volume(ct_array, threshold)
        cleaned = cleanup_metal_mask(mask)

        # Create output labelmap node
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "MetalSegmentation"
        )
        # Copy geometry from CT
        labelmap_node.CopyOrientation(ct_volume_node)

        # Use a volumes logic to copy the volume properties
        volumes_logic = slicer.modules.volumes.logic()
        volumes_logic.CreateLabelVolumeFromVolume(
            slicer.mrmlScene, labelmap_node, ct_volume_node
        )

        updateVolumeFromArray(labelmap_node, cleaned)
        return labelmap_node
```

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/metal_segmenter.py
git commit -m "feat: add Slicer wrapper for metal segmenter"
```

**Step 3: Merge branch**

```bash
git checkout main
git merge feat/metal-segmentation
git branch -d feat/metal-segmentation
```

---

## Branch 3: `feat/registration`

Implements CT-to-T1 rigid registration wrapper (Steps 2-3 in the design).

### Task 3.1: Registration wrapper

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/registration.py`

This is entirely Slicer-dependent (BRAINSFit CLI module), so no pytest-based tests. Integration testing will be done in Slicer.

**Step 1: Implement registration module**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/registration.py
"""CT-to-T1 rigid registration using BRAINSFit.

Example (in Slicer Python console)::

    reg = CTtoT1Registration()
    transform = reg.run(ct_node, t1_node)
"""

import slicer


class CTtoT1Registration:
    """Wraps BRAINSFit for rigid CT-to-T1 registration."""

    def create_rough_transform(self, ct_volume_node):
        """Create a linear transform node and apply it to the CT volume.

        The user will manually adjust this transform to roughly align CT to T1
        before running automated registration.

        Returns:
            vtkMRMLLinearTransformNode applied to the CT volume.
        """
        transform_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "CT_RoughAlignment"
        )
        ct_volume_node.SetAndObserveTransformNodeID(transform_node.GetID())
        return transform_node

    def run(
        self,
        ct_node,
        t1_node,
        initial_transform=None,
        sampling_percentage=0.02,
    ):
        """Run BRAINSFit rigid registration.

        Args:
            ct_node: Moving volume (post-implant CT).
            t1_node: Fixed volume (pre-implant T1 MRI).
            initial_transform: Optional initial transform from rough alignment.
            sampling_percentage: Fraction of voxels to sample (default 0.02).

        Returns:
            vtkMRMLLinearTransformNode with the registration result.
        """
        output_transform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "CT_to_T1_Registration"
        )

        params = {
            "fixedVolume": t1_node.GetID(),
            "movingVolume": ct_node.GetID(),
            "outputTransform": output_transform.GetID(),
            "useRigid": True,
            "samplingPercentage": sampling_percentage,
            "numberOfIterations": 1500,
            "minimumStepLength": 0.005,
            "numberOfHistogramBins": 50,
            "costMetric": "MMI",
        }

        if initial_transform is not None:
            params["initialTransform"] = initial_transform.GetID()

        cli_node = slicer.cli.runSync(
            slicer.modules.brainsfit, None, params
        )

        if cli_node.GetStatus() & cli_node.ErrorsMask:
            error_text = cli_node.GetErrorText()
            raise RuntimeError(f"BRAINSFit registration failed: {error_text}")

        # Apply result transform to CT
        ct_node.SetAndObserveTransformNodeID(output_transform.GetID())

        return output_transform

    def harden_transform(self, volume_node):
        """Harden the transform on a volume node (bake it into the volume)."""
        slicer.vtkSlicerTransformLogic().hardenTransform(volume_node)
```

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/registration.py
git commit -m "feat: add CT-to-T1 registration wrapper (BRAINSFit)"
```

**Step 3: Merge branch**

```bash
git checkout main
git merge feat/registration
git branch -d feat/registration
```

---

## Branch 4: `feat/electrode-detection`

The core detection algorithm. This is the most complex piece — heavy TDD.

### Task 4.1: Voxel extraction and clustering

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py`
- Create: `tests/test_electrode_detector.py`

**Step 1: Write test for voxel extraction and clustering**

```python
# tests/test_electrode_detector.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.electrode_detector import extract_metal_coords, cluster_into_electrodes


class TestExtractMetalCoords:
    def test_extracts_nonzero_coords(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2, 3, 4] = 1
        mask[5, 6, 7] = 1
        # Dummy spacing and origin
        coords = extract_metal_coords(mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        assert coords.shape == (2, 3)

    def test_applies_spacing_and_origin(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0, 0, 0] = 1
        coords = extract_metal_coords(mask, spacing=(2.0, 3.0, 4.0), origin=(10.0, 20.0, 30.0))
        np.testing.assert_allclose(coords[0], [10.0, 20.0, 30.0])


class TestClusterIntoElectrodes:
    def _make_line_cluster(self, start, direction, n_contacts, contact_voxels=3):
        """Create a synthetic electrode: a line of small clusters."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        points = []
        for i in range(n_contacts):
            center = np.array(start) + i * 3.5 * direction  # 3.5mm spacing
            # Small blob around each contact
            for _ in range(contact_voxels):
                jitter = np.random.randn(3) * 0.3
                points.append(center + jitter)
        return np.array(points)

    def test_two_separate_electrodes(self):
        np.random.seed(42)
        e1 = self._make_line_cluster([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_line_cluster([0, 50, 0], [0, 1, 0], 6)
        all_coords = np.vstack([e1, e2])
        clusters = cluster_into_electrodes(all_coords)
        assert len(clusters) == 2

    def test_single_electrode(self):
        np.random.seed(42)
        e1 = self._make_line_cluster([0, 0, 0], [1, 0, 0], 10)
        clusters = cluster_into_electrodes(e1)
        assert len(clusters) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_electrode_detector.py::TestExtractMetalCoords -v`
Expected: FAIL — module not found

**Step 3: Implement extraction and clustering**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
"""Automated electrode detection from metal segmentation.

Core algorithm functions operate on numpy arrays. The ElectrodeDetector class
wraps these for use with Slicer volume nodes.
"""

import numpy as np
from scipy import ndimage
from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams


def extract_metal_coords(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Convert binary mask voxels to RAS coordinates.

    Args:
        mask: Binary 3D array (IJK indexing).
        spacing: Voxel size in mm (I, J, K).
        origin: Volume origin in RAS.

    Returns:
        (N, 3) array of RAS coordinates.

    Example::

        coords = extract_metal_coords(mask, spacing=(0.5, 0.5, 0.5), origin=(0, 0, 0))
    """
    ijk = np.argwhere(mask > 0).astype(float)  # shape (N, 3)
    if len(ijk) == 0:
        return np.empty((0, 3))
    # Convert IJK to RAS (assuming RAS-aligned volume for simplicity;
    # full IJK-to-RAS transform handled in the Slicer wrapper)
    ras = ijk * np.array(spacing) + np.array(origin)
    return ras


def cluster_into_electrodes(
    coords: np.ndarray,
    distance_threshold: float = 10.0,
) -> list[np.ndarray]:
    """Group metal coordinates into electrode candidates using connected components.

    Uses a voxel-grid approach: discretize coords, find connected components,
    then return coordinate arrays per component.

    Args:
        coords: (N, 3) array of RAS coordinates.
        distance_threshold: Max distance (mm) between voxels to consider connected.

    Returns:
        List of (M, 3) arrays, one per electrode candidate.

    Example::

        clusters = cluster_into_electrodes(all_metal_coords)
    """
    if len(coords) == 0:
        return []

    # Discretize into a grid with cell size = distance_threshold
    cell_size = distance_threshold
    grid_coords = np.floor(coords / cell_size).astype(int)

    # Shift to non-negative indices
    grid_min = grid_coords.min(axis=0)
    grid_coords -= grid_min

    # Create binary volume
    grid_max = grid_coords.max(axis=0)
    grid_shape = tuple(grid_max + 1)
    grid = np.zeros(grid_shape, dtype=np.uint8)

    for gc in grid_coords:
        grid[tuple(gc)] = 1

    # Connected components on the grid
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled_grid, num_components = ndimage.label(grid, structure=struct)

    # Map each original point to its component
    labels = np.array([labeled_grid[tuple(gc)] for gc in grid_coords])

    clusters = []
    for comp_id in range(1, num_components + 1):
        cluster_mask = labels == comp_id
        clusters.append(coords[cluster_mask])

    return clusters
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_electrode_detector.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: add metal coord extraction and electrode clustering"
```

---

### Task 4.2: Line fitting and contact detection

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (add functions)
- Modify: `tests/test_electrode_detector.py` (add tests)

**Step 1: Write tests for line fitting and contact detection**

Append to `tests/test_electrode_detector.py`:

```python
from SEEGFellowLib.electrode_detector import fit_electrode_axis, detect_contacts_along_axis


class TestFitElectrodeAxis:
    def test_fits_line_along_x(self):
        np.random.seed(42)
        points = np.column_stack([
            np.linspace(0, 30, 50),
            np.random.randn(50) * 0.3,
            np.random.randn(50) * 0.3,
        ])
        center, direction = fit_electrode_axis(points)
        # Direction should be approximately along x-axis
        assert abs(abs(direction[0]) - 1.0) < 0.1

    def test_returns_unit_vector(self):
        points = np.column_stack([
            np.linspace(0, 10, 20),
            np.zeros(20),
            np.zeros(20),
        ])
        _, direction = fit_electrode_axis(points)
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-10)


class TestDetectContactsAlongAxis:
    def _make_contacts_1d(self, n_contacts, spacing, noise=0.2):
        """Create 1D density profile with peaks at regular spacing."""
        positions = []
        for i in range(n_contacts):
            center = i * spacing
            # Several points per contact
            for _ in range(5):
                positions.append(center + np.random.randn() * noise)
        return np.array(positions)

    def test_detects_evenly_spaced_contacts(self):
        np.random.seed(42)
        projections = self._make_contacts_1d(8, spacing=3.5)
        peaks = detect_contacts_along_axis(projections, expected_spacing=3.5)
        assert len(peaks) == 8

    def test_detects_gapped_contacts(self):
        """6 contacts - gap - 6 contacts."""
        np.random.seed(42)
        group1 = self._make_contacts_1d(6, spacing=3.5)
        gap = 6 * 3.5 + 10.0  # gap of 10mm after group 1 ends
        group2 = self._make_contacts_1d(6, spacing=3.5) + gap
        projections = np.concatenate([group1, group2])
        peaks = detect_contacts_along_axis(projections, expected_spacing=3.5)
        assert len(peaks) == 12
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_electrode_detector.py::TestFitElectrodeAxis -v`
Expected: FAIL — functions not found

**Step 3: Implement line fitting and contact detection**

Append to `electrode_detector.py`:

```python
def fit_electrode_axis(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line to electrode voxel positions via PCA.

    Args:
        coords: (N, 3) array of RAS coordinates.

    Returns:
        (center, direction): center is the mean point, direction is a unit vector
        along the first principal component.

    Example::

        center, direction = fit_electrode_axis(cluster_coords)
    """
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]  # first principal component
    direction /= np.linalg.norm(direction)
    return center, direction


def detect_contacts_along_axis(
    projections: np.ndarray,
    expected_spacing: float = 3.5,
    bin_width: float = 0.5,
    min_peak_height_fraction: float = 0.3,
) -> np.ndarray:
    """Detect contact positions from 1D projected metal voxel positions.

    Builds a density histogram along the electrode axis and finds peaks
    corresponding to individual contacts.

    Args:
        projections: 1D array of voxel positions projected onto the electrode axis.
        expected_spacing: Expected contact spacing in mm.
        bin_width: Histogram bin width in mm.
        min_peak_height_fraction: Minimum peak height as fraction of max peak.

    Returns:
        1D array of contact positions along the axis (sorted).

    Example::

        peaks = detect_contacts_along_axis(projected_positions, expected_spacing=3.5)
    """
    from scipy.signal import find_peaks

    if len(projections) == 0:
        return np.array([])

    # Build density histogram
    proj_min = projections.min() - expected_spacing
    proj_max = projections.max() + expected_spacing
    bins = np.arange(proj_min, proj_max + bin_width, bin_width)
    hist, bin_edges = np.histogram(projections, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks with minimum distance based on expected spacing
    min_distance = max(1, int(expected_spacing * 0.6 / bin_width))
    min_height = hist.max() * min_peak_height_fraction

    peaks_idx, _ = find_peaks(hist, distance=min_distance, height=min_height)
    peak_positions = bin_centers[peaks_idx]

    return np.sort(peak_positions)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_electrode_detector.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: add PCA line fitting and 1D contact peak detection"
```

---

### Task 4.3: Fragment merging and full detection pipeline

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (add functions)
- Modify: `tests/test_electrode_detector.py` (add tests)

**Step 1: Write tests for fragment merging and full pipeline**

Append to `tests/test_electrode_detector.py`:

```python
from SEEGFellowLib.electrode_detector import (
    merge_collinear_clusters,
    analyze_spacing,
    detect_electrodes,
)
from SEEGFellowLib.electrode_model import Electrode


class TestMergeCollinearClusters:
    def test_merges_two_collinear_fragments(self):
        """Two fragments along the same line should merge."""
        np.random.seed(42)
        frag1 = np.column_stack([
            np.linspace(0, 10, 20),
            np.random.randn(20) * 0.3,
            np.random.randn(20) * 0.3,
        ])
        frag2 = np.column_stack([
            np.linspace(20, 30, 20),  # same axis, gap in between
            np.random.randn(20) * 0.3,
            np.random.randn(20) * 0.3,
        ])
        clusters = [frag1, frag2]
        merged = merge_collinear_clusters(clusters, angle_tolerance=15.0)
        assert len(merged) == 1

    def test_does_not_merge_perpendicular(self):
        np.random.seed(42)
        frag1 = np.column_stack([
            np.linspace(0, 20, 30),
            np.zeros(30),
            np.zeros(30),
        ])
        frag2 = np.column_stack([
            np.zeros(30),
            np.linspace(40, 60, 30),  # perpendicular, far away
            np.zeros(30),
        ])
        clusters = [frag1, frag2]
        merged = merge_collinear_clusters(clusters, angle_tolerance=15.0)
        assert len(merged) == 2


class TestAnalyzeSpacing:
    def test_uniform_spacing(self):
        positions = np.array([0.0, 3.5, 7.0, 10.5, 14.0])
        spacing_info = analyze_spacing(positions)
        assert not spacing_info["has_gaps"]
        assert abs(spacing_info["contact_spacing"] - 3.5) < 0.5

    def test_gapped_spacing(self):
        # 4 contacts, gap, 4 contacts
        group1 = np.array([0.0, 3.5, 7.0, 10.5])
        group2 = group1 + 24.5  # gap of 14mm
        positions = np.concatenate([group1, group2])
        spacing_info = analyze_spacing(positions, gap_ratio_threshold=1.8)
        assert spacing_info["has_gaps"]


class TestDetectElectrodes:
    def _make_electrode(self, start, direction, n_contacts, spacing=3.5):
        """Create synthetic electrode voxels."""
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        points = []
        for i in range(n_contacts):
            center = np.array(start) + i * spacing * direction
            for _ in range(5):
                jitter = np.random.randn(3) * 0.3
                points.append(center + jitter)
        return np.array(points)

    def test_detects_two_electrodes(self):
        np.random.seed(42)
        e1 = self._make_electrode([0, 0, 0], [1, 0, 0], 8)
        e2 = self._make_electrode([0, 50, 0], [0, 1, 0], 6)
        all_coords = np.vstack([e1, e2])

        electrodes = detect_electrodes(all_coords)
        assert len(electrodes) == 2
        contact_counts = sorted([e.num_contacts for e in electrodes])
        assert contact_counts == [6, 8]

    def test_rejects_too_few_contacts(self):
        np.random.seed(42)
        e1 = self._make_electrode([0, 0, 0], [1, 0, 0], 2)  # only 2 contacts
        electrodes = detect_electrodes(e1, min_contacts=3)
        assert len(electrodes) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_electrode_detector.py::TestMergeCollinearClusters -v`
Expected: FAIL — functions not found

**Step 3: Implement fragment merging, spacing analysis, and full pipeline**

Append to `electrode_detector.py`:

```python
def merge_collinear_clusters(
    clusters: list[np.ndarray],
    angle_tolerance: float = 10.0,
    max_gap_mm: float = 30.0,
) -> list[np.ndarray]:
    """Merge clusters that share a collinear trajectory (fragments of a gapped electrode).

    Args:
        clusters: List of (N, 3) coordinate arrays.
        angle_tolerance: Maximum angle in degrees between axes to merge.
        max_gap_mm: Maximum distance between closest points of two clusters to merge.

    Returns:
        List of (M, 3) arrays after merging.

    Example::

        merged = merge_collinear_clusters(clusters, angle_tolerance=10.0)
    """
    if len(clusters) <= 1:
        return clusters

    # Compute axis for each cluster
    axes = []
    centers = []
    for cluster in clusters:
        if len(cluster) < 3:
            axes.append(None)
            centers.append(cluster.mean(axis=0))
        else:
            center, direction = fit_electrode_axis(cluster)
            axes.append(direction)
            centers.append(center)

    # Union-find for merging
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    angle_threshold_rad = np.radians(angle_tolerance)

    for i in range(len(clusters)):
        if axes[i] is None:
            continue
        for j in range(i + 1, len(clusters)):
            if axes[j] is None:
                continue
            # Check angle between axes
            cos_angle = abs(np.dot(axes[i], axes[j]))
            cos_angle = min(cos_angle, 1.0)
            angle = np.arccos(cos_angle)
            if angle > angle_threshold_rad:
                continue

            # Check distance: closest points between clusters
            # Use center-to-center distance as proxy (cheaper than full pairwise)
            dist = np.linalg.norm(centers[i] - centers[j])
            # Estimate extent of each cluster along its axis
            proj_i = np.dot(clusters[i] - centers[i], axes[i])
            proj_j = np.dot(clusters[j] - centers[j], axes[j])
            extent_i = proj_i.max() - proj_i.min()
            extent_j = proj_j.max() - proj_j.min()
            # Gap is center-to-center minus half-extents
            gap = dist - (extent_i + extent_j) / 2
            if gap < max_gap_mm:
                union(i, j)

    # Build merged clusters
    groups: dict[int, list[int]] = {}
    for i in range(len(clusters)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for indices in groups.values():
        merged.append(np.vstack([clusters[i] for i in indices]))

    return merged


def analyze_spacing(
    contact_positions: np.ndarray,
    gap_ratio_threshold: float = 1.8,
) -> dict:
    """Analyze inter-contact spacing to detect gaps.

    Args:
        contact_positions: Sorted 1D array of contact positions along axis.
        gap_ratio_threshold: If (long spacing / short spacing) exceeds this,
            classify as gapped electrode.

    Returns:
        Dict with keys: contact_spacing, has_gaps, gap_spacing, contacts_per_group.

    Example::

        info = analyze_spacing(np.array([0.0, 3.5, 7.0, 10.5, 24.5, 28.0]))
    """
    if len(contact_positions) < 2:
        return {
            "contact_spacing": 0.0,
            "has_gaps": False,
            "gap_spacing": None,
            "contacts_per_group": None,
        }

    spacings = np.diff(contact_positions)
    median_spacing = np.median(spacings)

    # Check for bimodal spacing
    short_mask = spacings < median_spacing * gap_ratio_threshold
    long_mask = ~short_mask

    if np.any(long_mask) and np.any(short_mask):
        short_spacings = spacings[short_mask]
        long_spacings = spacings[long_mask]
        ratio = np.median(long_spacings) / np.median(short_spacings)
        if ratio > gap_ratio_threshold:
            # Count contacts per group: find runs of short spacings
            gap_indices = np.where(long_mask)[0]
            group_sizes = []
            prev = 0
            for gi in gap_indices:
                group_sizes.append(gi - prev + 1)
                prev = gi + 1
            group_sizes.append(len(contact_positions) - prev)
            return {
                "contact_spacing": float(np.median(short_spacings)),
                "has_gaps": True,
                "gap_spacing": float(np.median(long_spacings)),
                "contacts_per_group": int(np.median(group_sizes)),
            }

    return {
        "contact_spacing": float(median_spacing),
        "has_gaps": False,
        "gap_spacing": None,
        "contacts_per_group": None,
    }


def orient_deepest_first(
    contact_positions: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient contacts so index 1 = deepest (closest to brain center at origin 0,0,0).

    The deepest contact is the one whose RAS position is closest to (0, 0, 0).

    Args:
        contact_positions: Sorted 1D array of positions along axis.
        axis_origin: 3D center of the electrode cluster.
        axis_direction: Unit vector along electrode axis.

    Returns:
        (sorted_positions, oriented_direction): positions sorted deepest-first,
        and direction pointing from deepest to most lateral.
    """
    # Compute RAS coordinates of first and last contact
    first_ras = axis_origin + contact_positions[0] * axis_direction
    last_ras = axis_origin + contact_positions[-1] * axis_direction

    # Deepest = closest to brain center (0,0,0)
    if np.linalg.norm(first_ras) > np.linalg.norm(last_ras):
        # Reverse: last is deeper
        return contact_positions[::-1] - contact_positions[-1], -axis_direction
    else:
        return contact_positions - contact_positions[0], axis_direction


def detect_electrodes(
    metal_coords: np.ndarray,
    min_contacts: int = 3,
    expected_spacing: float = 3.5,
    collinearity_tolerance: float = 10.0,
    gap_ratio_threshold: float = 1.8,
) -> list[Electrode]:
    """Full detection pipeline: cluster → fit → detect contacts → build Electrode objects.

    Args:
        metal_coords: (N, 3) array of all metal voxel RAS coordinates.
        min_contacts: Minimum contacts to accept an electrode candidate.
        expected_spacing: Expected contact spacing in mm.
        collinearity_tolerance: Max angle for merging collinear fragments.
        gap_ratio_threshold: Threshold for gap detection.

    Returns:
        List of Electrode objects with auto-numbered contacts (unnamed).

    Example::

        electrodes = detect_electrodes(all_metal_coords_ras)
    """
    # 1. Cluster
    clusters = cluster_into_electrodes(metal_coords)

    # 2. Merge collinear fragments (for gapped electrodes)
    clusters = merge_collinear_clusters(clusters, angle_tolerance=collinearity_tolerance)

    electrodes = []
    for cluster in clusters:
        if len(cluster) < 5:  # too few voxels to be an electrode
            continue

        # 3. Fit line
        center, direction = fit_electrode_axis(cluster)

        # 4. Project voxels onto axis
        projections = np.dot(cluster - center, direction)

        # 5. Detect contacts
        contact_positions = detect_contacts_along_axis(
            projections, expected_spacing=expected_spacing
        )

        if len(contact_positions) < min_contacts:
            continue

        # 6. Analyze spacing
        spacing_info = analyze_spacing(contact_positions, gap_ratio_threshold)

        # 7. Orient deepest first
        sorted_positions, oriented_dir = orient_deepest_first(
            contact_positions, center, direction
        )

        # 8. Build Electrode
        params = ElectrodeParams(
            contact_length=2.0,  # default, user can adjust later
            contact_spacing=spacing_info["contact_spacing"],
            contact_diameter=0.8,
            gap_spacing=spacing_info["gap_spacing"],
            contacts_per_group=spacing_info["contacts_per_group"],
        )

        contacts = []
        for i, pos in enumerate(sorted_positions):
            ras = center + pos * oriented_dir
            contacts.append(
                Contact(index=i + 1, position_ras=tuple(ras))
            )

        electrode = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=tuple(oriented_dir),
        )
        electrodes.append(electrode)

    return electrodes
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_electrode_detector.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py tests/test_electrode_detector.py
git commit -m "feat: add fragment merging, spacing analysis, and full detection pipeline"
```

---

### Task 4.4: Slicer wrapper for electrode detection

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py` (append class)

**Step 1: Add ElectrodeDetector class**

Append to `electrode_detector.py`:

```python
class ElectrodeDetector:
    """Slicer wrapper: detects all electrodes from a metal segmentation volume.

    Example (in Slicer Python console)::

        detector = ElectrodeDetector()
        electrodes = detector.detect_all(metal_labelmap_node, ct_volume_node)
    """

    def __init__(
        self,
        min_contacts: int = 3,
        expected_spacing: float = 3.5,
        collinearity_tolerance: float = 10.0,
        gap_ratio_threshold: float = 1.8,
    ):
        self.min_contacts = min_contacts
        self.expected_spacing = expected_spacing
        self.collinearity_tolerance = collinearity_tolerance
        self.gap_ratio_threshold = gap_ratio_threshold

    def detect_all(self, metal_volume_node, ct_volume_node) -> list[Electrode]:
        """From a binary metal segmentation, find all electrodes and their contacts.

        Args:
            metal_volume_node: vtkMRMLLabelMapVolumeNode with metal mask.
            ct_volume_node: vtkMRMLScalarVolumeNode (used for geometry info).

        Returns:
            List of Electrode objects.
        """
        from slicer.util import arrayFromVolume

        mask = arrayFromVolume(metal_volume_node)

        # Get IJK-to-RAS transform from the volume node
        ijk_to_ras = self._get_ijk_to_ras_matrix(metal_volume_node)

        # Extract coordinates and transform to RAS
        ijk_coords = np.argwhere(mask > 0).astype(float)
        if len(ijk_coords) == 0:
            return []

        # Apply full IJK-to-RAS (handles non-axis-aligned volumes)
        ones = np.ones((len(ijk_coords), 1))
        ijk_h = np.hstack([ijk_coords, ones])  # homogeneous
        ras_h = (ijk_to_ras @ ijk_h.T).T
        ras_coords = ras_h[:, :3]

        return detect_electrodes(
            ras_coords,
            min_contacts=self.min_contacts,
            expected_spacing=self.expected_spacing,
            collinearity_tolerance=self.collinearity_tolerance,
            gap_ratio_threshold=self.gap_ratio_threshold,
        )

    @staticmethod
    def _get_ijk_to_ras_matrix(volume_node) -> np.ndarray:
        """Get 4x4 IJK-to-RAS matrix from a volume node."""
        import vtk

        mat = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(mat)
        return np.array(
            [[mat.GetElement(i, j) for j in range(4)] for i in range(4)]
        )
```

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_detector.py
git commit -m "feat: add Slicer wrapper for electrode detector"
```

**Step 3: Merge branch**

```bash
git checkout main
git merge feat/electrode-detection
git branch -d feat/electrode-detection
```

---

## Branch 5: `feat/trajectory-detector`

Fallback: single-electrode detection from a seed point (Step 8 in the design).

### Task 5.1: Trajectory detector with tests

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/trajectory_detector.py`
- Create: `tests/test_trajectory_detector.py`

**Step 1: Write tests**

```python
# tests/test_trajectory_detector.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.trajectory_detector import (
    estimate_trajectory,
    detect_contacts_from_intensity_profile,
)
from SEEGFellowLib.electrode_model import ElectrodeParams


class TestEstimateTrajectory:
    def test_pca_direction(self):
        """PCA on a line of points along x should give x-direction."""
        np.random.seed(42)
        n = 50
        points = np.column_stack([
            np.linspace(0, 20, n),
            np.random.randn(n) * 0.3,
            np.random.randn(n) * 0.3,
        ])
        direction = estimate_trajectory(points)
        assert abs(abs(direction[0]) - 1.0) < 0.1

    def test_direction_hint(self):
        """When given a direction hint, should use it."""
        np.random.seed(42)
        points = np.random.randn(20, 3) * 5  # random blob
        hint = np.array([0.0, 1.0, 0.0])
        direction = estimate_trajectory(points, direction_hint=hint)
        # Should be close to hint
        assert abs(np.dot(direction, hint)) > 0.9


class TestDetectContactsFromProfile:
    def _make_profile(self, n_contacts, spacing, contact_width=1.5, noise=50.0):
        """Create a synthetic intensity profile along an electrode."""
        length = (n_contacts + 1) * spacing
        positions = np.arange(0, length, 0.2)
        intensities = np.full_like(positions, 200.0)  # background
        # Add peaks at contact positions
        for i in range(n_contacts):
            center = (i + 0.5) * spacing
            mask = np.abs(positions - center) < contact_width / 2
            intensities[mask] = 3000.0
        intensities += np.random.randn(len(intensities)) * noise
        return positions, intensities

    def test_detects_correct_number(self):
        np.random.seed(42)
        positions, intensities = self._make_profile(8, spacing=3.5)
        contacts = detect_contacts_from_intensity_profile(
            positions, intensities, num_contacts=8, expected_spacing=3.5
        )
        assert len(contacts) == 8

    def test_contacts_are_sorted(self):
        np.random.seed(42)
        positions, intensities = self._make_profile(6, spacing=3.5)
        contacts = detect_contacts_from_intensity_profile(
            positions, intensities, num_contacts=6, expected_spacing=3.5
        )
        for i in range(len(contacts) - 1):
            assert contacts[i] < contacts[i + 1]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_detector.py -v`
Expected: FAIL

**Step 3: Implement core functions**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/trajectory_detector.py
"""Single-electrode detection from a seed point (manual fallback).

Core functions operate on numpy arrays. The IntensityProfileDetector class
wraps these for Slicer.
"""

import numpy as np
from scipy.signal import find_peaks
from SEEGFellowLib.electrode_model import Contact, ElectrodeParams


def estimate_trajectory(
    metal_coords: np.ndarray,
    direction_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate electrode trajectory from nearby metal voxels.

    Args:
        metal_coords: (N, 3) array of metal voxel coordinates.
        direction_hint: Optional unit vector hint for the trajectory direction.

    Returns:
        Unit vector along the electrode trajectory.

    Example::

        direction = estimate_trajectory(nearby_metal_coords)
    """
    if direction_hint is not None:
        hint = np.array(direction_hint, dtype=float)
        return hint / np.linalg.norm(hint)

    # PCA
    centered = metal_coords - metal_coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction /= np.linalg.norm(direction)
    return direction


def detect_contacts_from_intensity_profile(
    positions: np.ndarray,
    intensities: np.ndarray,
    num_contacts: int,
    expected_spacing: float = 3.5,
    snap_tolerance: float = 1.0,
) -> np.ndarray:
    """Detect contact positions from an intensity profile along the trajectory.

    Args:
        positions: 1D array of sample positions along trajectory (mm).
        intensities: 1D array of CT intensities at each position.
        num_contacts: Expected number of contacts.
        expected_spacing: Expected center-to-center spacing (mm).
        snap_tolerance: If peaks are within this distance of the expected
            grid, snap to the grid.

    Returns:
        Sorted 1D array of contact positions along the trajectory.

    Example::

        contacts = detect_contacts_from_intensity_profile(
            positions, intensities, num_contacts=8, expected_spacing=3.5
        )
    """
    step = positions[1] - positions[0] if len(positions) > 1 else 0.2
    min_distance = max(1, int(expected_spacing * 0.6 / step))

    peaks_idx, properties = find_peaks(
        intensities,
        distance=min_distance,
        height=np.median(intensities),
    )

    if len(peaks_idx) == 0:
        return np.array([])

    peak_positions = positions[peaks_idx]
    peak_heights = properties["peak_heights"]

    # Take the num_contacts strongest peaks
    if len(peak_positions) > num_contacts:
        top_indices = np.argsort(peak_heights)[-num_contacts:]
        peak_positions = np.sort(peak_positions[top_indices])
    else:
        peak_positions = np.sort(peak_positions)

    # Snap to expected grid if close enough
    if len(peak_positions) >= 2:
        grid_start = peak_positions[0]
        expected_grid = grid_start + np.arange(len(peak_positions)) * expected_spacing
        deviations = np.abs(peak_positions - expected_grid)
        if np.all(deviations < snap_tolerance):
            peak_positions = expected_grid

    return peak_positions
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_detector.py -v`
Expected: all PASS

**Step 5: Add IntensityProfileDetector Slicer wrapper**

Append to `trajectory_detector.py`:

```python
class IntensityProfileDetector:
    """Slicer wrapper: detects contacts along a single electrode from a seed point.

    Example (in Slicer Python console)::

        detector = IntensityProfileDetector()
        contacts = detector.detect(
            seed_ras=(10.0, 20.0, 30.0),
            ct_volume_node=ct_node,
            num_contacts=8,
            params=ElectrodeParams(contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8),
        )
    """

    def detect(
        self,
        seed_ras: tuple[float, float, float],
        ct_volume_node,
        num_contacts: int,
        params: ElectrodeParams,
        direction_hint: tuple[float, float, float] | None = None,
        search_radius: float = 20.0,
        metal_threshold: float = 2500,
    ) -> list[Contact]:
        """Given a seed point, find contacts along an electrode.

        Args:
            seed_ras: (R, A, S) coordinates of the deepest contact.
            ct_volume_node: Slicer CT volume node.
            num_contacts: Expected number of contacts.
            params: Electrode physical parameters.
            direction_hint: Optional (R, A, S) direction hint.
            search_radius: Radius in mm for local neighborhood.
            metal_threshold: HU threshold for metal.

        Returns:
            List of Contact objects, numbered 1 (deepest) to num_contacts.
        """
        import slicer
        from slicer.util import arrayFromVolume
        import vtk

        seed = np.array(seed_ras)
        ct_array = arrayFromVolume(ct_volume_node)

        # Get RAS-to-IJK matrix
        ras_to_ijk = vtk.vtkMatrix4x4()
        ct_volume_node.GetRASToIJKMatrix(ras_to_ijk)

        ijk_to_ras = vtk.vtkMatrix4x4()
        ct_volume_node.GetIJKToRASMatrix(ijk_to_ras)

        # Convert seed to IJK
        seed_h = [*seed_ras, 1.0]
        seed_ijk = [0.0] * 4
        ras_to_ijk.MultiplyPoint(seed_h, seed_ijk)
        seed_ijk = np.array(seed_ijk[:3])

        # Get spacing for radius calculation
        spacing = np.array(ct_volume_node.GetSpacing())

        # Extract local neighborhood
        radius_voxels = (search_radius / spacing).astype(int)
        slices = []
        for dim in range(3):
            lo = max(0, int(seed_ijk[dim]) - radius_voxels[dim])
            hi = min(ct_array.shape[dim], int(seed_ijk[dim]) + radius_voxels[dim] + 1)
            slices.append(slice(lo, hi))

        local_ct = ct_array[slices[0], slices[1], slices[2]]

        # Threshold to isolate metal
        metal_mask = local_ct >= metal_threshold
        metal_ijk = np.argwhere(metal_mask).astype(float)

        if len(metal_ijk) == 0:
            return []

        # Convert local IJK back to RAS
        offset = np.array([s.start for s in slices])
        metal_ijk_global = metal_ijk + offset
        metal_ras = np.array([
            [
                sum(ijk_to_ras.GetElement(r, c) * (metal_ijk_global[j, c] if c < 3 else 1.0)
                    for c in range(4))
                for r in range(3)
            ]
            for j in range(len(metal_ijk_global))
        ])

        # Estimate trajectory
        hint = np.array(direction_hint) if direction_hint else None
        direction = estimate_trajectory(metal_ras, direction_hint=hint)

        # Orient direction away from brain center
        outward = seed + direction * 10
        inward = seed - direction * 10
        if np.linalg.norm(outward) < np.linalg.norm(inward):
            direction = -direction

        # Sample intensity profile along trajectory
        profile_length = (num_contacts - 1) * params.contact_spacing + params.contact_spacing * 2
        sample_positions = np.arange(0, profile_length, 0.2)
        sample_ras = seed[:, None] + direction[:, None] * sample_positions[None, :]
        sample_ras = sample_ras.T  # (N, 3)

        # Convert sample points to IJK and interpolate
        from scipy.ndimage import map_coordinates

        sample_ijk = np.array([
            [
                sum(ras_to_ijk.GetElement(r, c) * (sample_ras[j, c] if c < 3 else 1.0)
                    for c in range(4))
                for r in range(3)
            ]
            for j in range(len(sample_ras))
        ])

        intensities = map_coordinates(
            ct_array.astype(float), sample_ijk.T, order=1, mode="constant", cval=0
        )

        # Detect contacts
        contact_offsets = detect_contacts_from_intensity_profile(
            sample_positions, intensities, num_contacts, params.contact_spacing
        )

        contacts = []
        for i, offset in enumerate(contact_offsets):
            pos = seed + direction * offset
            contacts.append(
                Contact(index=i + 1, position_ras=tuple(pos))
            )

        return contacts
```

**Step 6: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/trajectory_detector.py tests/test_trajectory_detector.py
git commit -m "feat: add single-electrode trajectory detector (seed-point fallback)"
```

**Step 7: Merge branch**

```bash
git checkout main
git merge feat/trajectory-detector
git branch -d feat/trajectory-detector
```

---

## Branch 6: `feat/contact-segmenter`

Creates per-contact cylindrical segments for 3D visualization (Step 9 in the design).

### Task 6.1: Contact segmenter

**Files:**
- Create: `SEEGFellow/SEEGFellow/SEEGFellowLib/contact_segmenter.py`
- Create: `tests/test_contact_segmenter.py`

**Step 1: Write tests for cylinder voxel generation**

```python
# tests/test_contact_segmenter.py
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

import numpy as np
from SEEGFellowLib.contact_segmenter import generate_cylinder_mask


class TestGenerateCylinderMask:
    def test_creates_nonzero_mask(self):
        center = np.array([15.0, 15.0, 15.0])
        direction = np.array([1.0, 0.0, 0.0])
        mask = generate_cylinder_mask(
            shape=(30, 30, 30),
            center=center,
            direction=direction,
            length=2.0,
            diameter=0.8,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        assert np.any(mask > 0)

    def test_cylinder_volume_reasonable(self):
        """Volume of filled voxels should approximate pi*r^2*L."""
        center = np.array([25.0, 25.0, 25.0])
        direction = np.array([0.0, 0.0, 1.0])
        length = 4.0
        diameter = 2.0
        spacing = (0.5, 0.5, 0.5)
        mask = generate_cylinder_mask(
            shape=(100, 100, 100),
            center=center,
            direction=direction,
            length=length,
            diameter=diameter,
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
        )
        voxel_vol = 0.5 ** 3
        filled_vol = np.sum(mask) * voxel_vol
        expected_vol = np.pi * (diameter / 2) ** 2 * length
        # Should be within 50% (discretization error)
        assert abs(filled_vol - expected_vol) / expected_vol < 0.5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contact_segmenter.py -v`
Expected: FAIL

**Step 3: Implement cylinder mask and Slicer wrapper**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/contact_segmenter.py
"""Creates a Slicer segmentation with per-contact cylindrical segments.

Example (in Slicer Python console)::

    segmenter = ContactSegmenter()
    seg_node = segmenter.create_segmentation(electrodes, ct_volume_node)
"""

import numpy as np
from SEEGFellowLib.electrode_model import Electrode


# Distinct colors for electrodes (RGB, 0-1)
ELECTRODE_COLORS = [
    (0.9, 0.2, 0.2),  # red
    (0.2, 0.6, 0.9),  # blue
    (0.2, 0.9, 0.3),  # green
    (0.9, 0.7, 0.1),  # yellow
    (0.7, 0.2, 0.9),  # purple
    (0.9, 0.5, 0.1),  # orange
    (0.1, 0.9, 0.8),  # cyan
    (0.9, 0.2, 0.6),  # pink
]


def generate_cylinder_mask(
    shape: tuple[int, int, int],
    center: np.ndarray,
    direction: np.ndarray,
    length: float,
    diameter: float,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Generate a binary mask of a cylinder in a volume grid.

    Args:
        shape: Volume dimensions (I, J, K).
        center: Cylinder center in RAS coordinates.
        direction: Unit vector along cylinder axis.
        length: Cylinder length in mm.
        diameter: Cylinder diameter in mm.
        spacing: Voxel spacing (I, J, K) in mm.
        origin: Volume origin in RAS.

    Returns:
        Binary uint8 array of shape `shape`.

    Example::

        mask = generate_cylinder_mask(
            shape=(100, 100, 100),
            center=np.array([50.0, 50.0, 50.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            length=2.0, diameter=0.8,
            spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0),
        )
    """
    radius = diameter / 2.0
    half_length = length / 2.0
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    mask = np.zeros(shape, dtype=np.uint8)

    # Create grid of RAS coordinates
    ii, jj, kk = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    ras_coords = np.stack(
        [
            ii * spacing[0] + origin[0],
            jj * spacing[1] + origin[1],
            kk * spacing[2] + origin[2],
        ],
        axis=-1,
    )  # shape (I, J, K, 3)

    # Vector from center to each voxel
    diff = ras_coords - center  # (I, J, K, 3)

    # Project onto axis
    along_axis = np.einsum("ijkd,d->ijk", diff, direction)

    # Perpendicular distance
    proj_on_axis = along_axis[..., None] * direction  # (I, J, K, 3)
    perp = diff - proj_on_axis
    perp_dist = np.linalg.norm(perp, axis=-1)

    # Inside cylinder
    inside = (np.abs(along_axis) <= half_length) & (perp_dist <= radius)
    mask[inside] = 1

    return mask


class ContactSegmenter:
    """Creates a Slicer segmentation with per-contact cylindrical segments."""

    def create_segmentation(self, electrodes: list[Electrode], reference_volume_node):
        """Create a segmentation node with one segment per contact.

        Args:
            electrodes: List of Electrode objects with contacts.
            reference_volume_node: Volume node for geometry reference.

        Returns:
            vtkMRMLSegmentationNode.
        """
        import slicer
        import vtk

        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "ElectrodeContacts"
        )
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume_node)

        segmentation = seg_node.GetSegmentation()

        for electrode_idx, electrode in enumerate(electrodes):
            color = ELECTRODE_COLORS[electrode_idx % len(ELECTRODE_COLORS)]

            for contact in electrode.contacts:
                segment_id = segmentation.AddEmptySegment(
                    contact.label, contact.label
                )
                segment = segmentation.GetSegment(segment_id)
                segment.SetColor(*color)

                # Create cylinder labelmap
                # For efficiency, we create a small labelmap around the contact
                # and add it as a binary labelmap representation
                # This is done via Slicer's segmentation editing utilities

        # Generate closed surface for 3D visualization
        seg_node.CreateClosedSurfaceRepresentation()

        return seg_node
```

Note: The full Slicer segment creation (converting cylinder masks into segment binary labelmaps) requires using Slicer's `vtkOrientedImageData` API. The `create_segmentation` method above is a skeleton — the full implementation uses `slicer.vtkSlicerSegmentationsModuleLogic` to import labelmaps. This will be completed during integration testing in Slicer.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contact_segmenter.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/contact_segmenter.py tests/test_contact_segmenter.py
git commit -m "feat: add contact segmenter (cylinder mask generation)"
```

**Step 6: Merge branch**

```bash
git checkout main
git merge feat/contact-segmenter
git branch -d feat/contact-segmenter
```

---

## Branch 7: `feat/gui`

The wizard-style GUI and widget that wires all pipeline steps together.

### Task 7.1: Create the UI file

**Files:**
- Create: `SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui`

**Step 1: Write the Qt Designer UI file**

Create a `.ui` file with collapsible buttons for each step. The UI contains:

1. **Load Data** section: two `ctkPathLineEdit` widgets for T1 and CT, a "Load" button
2. **Rough Alignment** section: instruction label, "Create Transform" button, "Open Transforms Module" button, "Done" button
3. **Co-registration** section: "Register" button, progress bar, "Accept"/"Re-run" buttons
4. **Metal Segmentation** section: `ctkSliderWidget` for threshold, "Segment" button, "Accept" button
5. **Electrode Detection** section: "Detect All" button, `QTableWidget` for results, "Apply Names" button
6. **Manual Fallback** section (collapsed): preset dropdown, parameter fields, "Place Seed"/"Detect" buttons
7. **Results & Export** section: `QTableWidget` for contacts, "Create Segmentation"/"Export CSV" buttons
8. **Electrode List** section: `QListWidget`, "Delete Electrode" button

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>SEEGFellow</class>
 <widget class="qMRMLWidget" name="SEEGFellow">
  <layout class="QVBoxLayout" name="verticalLayout">

   <!-- Step 1: Load Data -->
   <item>
    <widget class="ctkCollapsibleButton" name="loadDataCollapsibleButton">
     <property name="text"><string>Step 1: Load Data</string></property>
     <layout class="QFormLayout" name="loadDataFormLayout">
      <item row="0" column="0"><widget class="QLabel"><property name="text"><string>T1 MRI:</string></property></widget></item>
      <item row="0" column="1"><widget class="ctkPathLineEdit" name="t1PathLineEdit"/></item>
      <item row="1" column="0"><widget class="QLabel"><property name="text"><string>Post-implant CT:</string></property></widget></item>
      <item row="1" column="1"><widget class="ctkPathLineEdit" name="ctPathLineEdit"/></item>
      <item row="2" column="1"><widget class="QPushButton" name="loadButton"><property name="text"><string>Load Volumes</string></property></widget></item>
     </layout>
    </widget>
   </item>

   <!-- Step 2: Rough Alignment -->
   <item>
    <widget class="ctkCollapsibleButton" name="roughAlignCollapsibleButton">
     <property name="text"><string>Step 2: Rough Alignment</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="roughAlignLayout">
      <item><widget class="QLabel" name="roughAlignInstructions"><property name="text"><string>Use the Transforms module to roughly align the CT to the T1. Translate and rotate until the CT skull roughly overlaps the T1.</string></property><property name="wordWrap"><bool>true</bool></property></widget></item>
      <item><widget class="QPushButton" name="createTransformButton"><property name="text"><string>Create Transform</string></property></widget></item>
      <item><widget class="QPushButton" name="openTransformsButton"><property name="text"><string>Open Transforms Module</string></property></widget></item>
      <item><widget class="QPushButton" name="roughAlignDoneButton"><property name="text"><string>Done — Proceed to Registration</string></property></widget></item>
     </layout>
    </widget>
   </item>

   <!-- Step 3: Co-registration -->
   <item>
    <widget class="ctkCollapsibleButton" name="registrationCollapsibleButton">
     <property name="text"><string>Step 3: Co-registration</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="registrationLayout">
      <item><widget class="QPushButton" name="registerButton"><property name="text"><string>Register (BRAINSFit Rigid)</string></property></widget></item>
      <item><widget class="QProgressBar" name="registrationProgressBar"><property name="value"><number>0</number></property></widget></item>
      <item>
       <layout class="QHBoxLayout">
        <item><widget class="QPushButton" name="acceptRegistrationButton"><property name="text"><string>Accept</string></property></widget></item>
        <item><widget class="QPushButton" name="rerunRegistrationButton"><property name="text"><string>Re-run</string></property></widget></item>
       </layout>
      </item>
     </layout>
    </widget>
   </item>

   <!-- Step 4: Metal Segmentation -->
   <item>
    <widget class="ctkCollapsibleButton" name="metalSegCollapsibleButton">
     <property name="text"><string>Step 4: Metal Segmentation</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="metalSegLayout">
      <item>
       <layout class="QFormLayout">
        <item row="0" column="0"><widget class="QLabel"><property name="text"><string>HU Threshold:</string></property></widget></item>
        <item row="0" column="1"><widget class="ctkSliderWidget" name="thresholdSlider"><property name="minimum"><double>500</double></property><property name="maximum"><double>4000</double></property><property name="value"><double>2500</double></property><property name="singleStep"><double>50</double></property></widget></item>
       </layout>
      </item>
      <item><widget class="QPushButton" name="segmentMetalButton"><property name="text"><string>Segment Metal</string></property></widget></item>
      <item>
       <layout class="QHBoxLayout">
        <item><widget class="QPushButton" name="acceptSegmentationButton"><property name="text"><string>Accept</string></property></widget></item>
        <item><widget class="QPushButton" name="adjustSegmentationButton"><property name="text"><string>Adjust &amp; Re-run</string></property></widget></item>
       </layout>
      </item>
     </layout>
    </widget>
   </item>

   <!-- Step 5: Electrode Detection -->
   <item>
    <widget class="ctkCollapsibleButton" name="detectionCollapsibleButton">
     <property name="text"><string>Step 5: Electrode Detection</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="detectionLayout">
      <item><widget class="QPushButton" name="detectElectrodesButton"><property name="text"><string>Detect All Electrodes</string></property></widget></item>
      <item><widget class="QTableWidget" name="electrodeTable"/></item>
      <item><widget class="QPushButton" name="applyNamesButton"><property name="text"><string>Apply Names</string></property></widget></item>
     </layout>
    </widget>
   </item>

   <!-- Step 6: Manual Fallback -->
   <item>
    <widget class="ctkCollapsibleButton" name="manualFallbackCollapsibleButton">
     <property name="text"><string>Manual Fallback (Seed-Point Mode)</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QFormLayout" name="manualFallbackLayout">
      <item row="0" column="0"><widget class="QLabel"><property name="text"><string>Num. contacts:</string></property></widget></item>
      <item row="0" column="1"><widget class="QSpinBox" name="numContactsSpinBox"><property name="minimum"><number>1</number></property><property name="maximum"><number>20</number></property><property name="value"><number>8</number></property></widget></item>
      <item row="1" column="0"><widget class="QLabel"><property name="text"><string>Spacing (mm):</string></property></widget></item>
      <item row="1" column="1"><widget class="QDoubleSpinBox" name="spacingSpinBox"><property name="minimum"><double>1.0</double></property><property name="maximum"><double>10.0</double></property><property name="value"><double>3.5</double></property><property name="singleStep"><double>0.1</double></property></widget></item>
      <item row="2" column="1"><widget class="QPushButton" name="placeSeedButton"><property name="text"><string>Place Seed Point</string></property></widget></item>
      <item row="3" column="1"><widget class="QPushButton" name="placeDirectionButton"><property name="text"><string>Set Direction Hint (optional)</string></property></widget></item>
      <item row="4" column="1"><widget class="QPushButton" name="detectSingleButton"><property name="text"><string>Detect Contacts</string></property></widget></item>
     </layout>
    </widget>
   </item>

   <!-- Step 7: Review & Export -->
   <item>
    <widget class="ctkCollapsibleButton" name="exportCollapsibleButton">
     <property name="text"><string>Results &amp; Export</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="exportLayout">
      <item><widget class="QTableWidget" name="contactTable"/></item>
      <item><widget class="QPushButton" name="createSegmentationButton"><property name="text"><string>Create Segmentation</string></property></widget></item>
      <item><widget class="QPushButton" name="exportCsvButton"><property name="text"><string>Export CSV</string></property></widget></item>
     </layout>
    </widget>
   </item>

   <!-- Electrode List -->
   <item>
    <widget class="ctkCollapsibleButton" name="electrodeListCollapsibleButton">
     <property name="text"><string>Electrode List</string></property>
     <property name="collapsed"><bool>true</bool></property>
     <layout class="QVBoxLayout" name="electrodeListLayout">
      <item><widget class="QListWidget" name="electrodeListWidget"/></item>
      <item><widget class="QPushButton" name="deleteElectrodeButton"><property name="text"><string>Delete Selected Electrode</string></property></widget></item>
     </layout>
    </widget>
   </item>

   <item><spacer name="verticalSpacer"><property name="orientation"><enum>Qt::Vertical</enum></property></spacer></item>
  </layout>
 </widget>
</ui>
```

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/Resources/UI/SEEGFellow.ui
git commit -m "feat: add wizard-style UI layout"
```

---

### Task 7.2: Implement SEEGFellowWidget

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellow.py` (rewrite Widget and Logic classes)

This is the largest single task. The widget wires UI elements to the logic pipeline.

**Step 1: Implement the full module file**

Rewrite `SEEGFellow.py` with the complete Widget and Logic classes. The widget:

- Loads the `.ui` file
- Connects buttons to logic methods
- Manages step progression (enable/disable sections)
- Populates tables with detection results
- Handles fiducial creation and interaction

Key logic methods:
- `load_volumes(t1_path, ct_path)` — loads volumes into Slicer
- `create_rough_transform()` — creates transform for manual alignment
- `run_registration()` — calls `CTtoT1Registration.run()`
- `run_metal_segmentation(threshold)` — calls `MetalSegmenter.segment()`
- `run_electrode_detection()` — calls `ElectrodeDetector.detect_all()`
- `run_single_detection(seed, num_contacts, params)` — calls `IntensityProfileDetector.detect()`
- `create_fiducials(electrodes)` — creates markup fiducial nodes
- `export_csv(path)` — exports contact coordinates
- `create_segmentation()` — calls `ContactSegmenter.create_segmentation()`

The full implementation of this file should follow the patterns established in the design document (section "GUI Design"). Each button's `clicked` signal connects to a method that:
1. Runs the relevant algorithm
2. Updates the display
3. Expands the next step's collapsible button

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellow.py
git commit -m "feat: implement SEEGFellowWidget with full wizard pipeline"
```

---

### Task 7.3: Update `__init__.py` and CMakeLists

**Files:**
- Modify: `SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py`
- Verify: `SEEGFellow/SEEGFellow/CMakeLists.txt` includes all files

**Step 1: Update `__init__.py`**

```python
# SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py
from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams
from SEEGFellowLib.metal_segmenter import MetalSegmenter
from SEEGFellowLib.registration import CTtoT1Registration
from SEEGFellowLib.electrode_detector import ElectrodeDetector
from SEEGFellowLib.trajectory_detector import IntensityProfileDetector
from SEEGFellowLib.contact_segmenter import ContactSegmenter
```

**Step 2: Verify CMakeLists includes UI file**

Add to `MODULE_PYTHON_RESOURCES` in `SEEGFellow/SEEGFellow/CMakeLists.txt`:

```cmake
set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )
```

**Step 3: Commit**

```bash
git add SEEGFellow/SEEGFellow/SEEGFellowLib/__init__.py SEEGFellow/SEEGFellow/CMakeLists.txt
git commit -m "chore: update imports and build config"
```

**Step 4: Merge branch**

```bash
git checkout main
git merge feat/gui
git branch -d feat/gui
```

---

## Branch 8: `feat/integration-testing`

End-to-end integration testing inside Slicer.

### Task 8.1: Write Slicer integration test

**Files:**
- Modify: `SEEGFellow/SEEGFellow/Testing/Python/SEEGFellowTest.py`

**Step 1: Write integration test class**

```python
# SEEGFellow/SEEGFellow/Testing/Python/SEEGFellowTest.py
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest


class SEEGFellowTest(ScriptedLoadableModuleTest):
    """Integration tests for SEEGFellow — run inside Slicer."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_module_loads()
        self.test_metal_segmenter_with_synthetic_volume()
        self.test_electrode_detector_with_synthetic_data()

    def test_module_loads(self):
        """Verify the module loads without error."""
        self.assertIsNotNone(slicer.modules.seegfellow)

    def test_metal_segmenter_with_synthetic_volume(self):
        """Create a synthetic CT volume with known metal and verify segmentation."""
        import numpy as np

        # Create synthetic CT volume
        vol_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "TestCT"
        )
        vol_node.SetSpacing(0.5, 0.5, 0.5)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(60, 60, 60)
        image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        vol_node.SetAndObserveImageData(image_data)

        # Fill with background
        arr = slicer.util.arrayFromVolume(vol_node)
        arr[:] = 200.0
        # Add metal blob
        arr[25:35, 30, 30] = 3000.0
        slicer.util.arrayFromVolumeModified(vol_node)

        from SEEGFellowLib.metal_segmenter import MetalSegmenter
        segmenter = MetalSegmenter()
        result = segmenter.segment(vol_node, threshold=2500)

        result_arr = slicer.util.arrayFromVolume(result)
        self.assertTrue(result_arr[30, 30, 30] > 0, "Metal voxel should be segmented")
        self.assertEqual(result_arr[5, 5, 5], 0, "Background should not be segmented")

    def test_electrode_detector_with_synthetic_data(self):
        """Test electrode detection with synthetic metal coordinates."""
        import numpy as np
        from SEEGFellowLib.electrode_detector import detect_electrodes

        # Create a synthetic electrode: 8 contacts along x-axis
        np.random.seed(42)
        coords = []
        for i in range(8):
            center = np.array([i * 3.5, 0.0, 0.0])
            for _ in range(10):
                coords.append(center + np.random.randn(3) * 0.3)
        coords = np.array(coords)

        electrodes = detect_electrodes(coords)
        self.assertEqual(len(electrodes), 1, f"Expected 1 electrode, got {len(electrodes)}")
        self.assertEqual(electrodes[0].num_contacts, 8, f"Expected 8 contacts, got {electrodes[0].num_contacts}")
```

**Step 2: Commit**

```bash
git add SEEGFellow/SEEGFellow/Testing/Python/SEEGFellowTest.py
git commit -m "test: add Slicer integration tests"
```

### Task 8.2: Run all tests and fix issues

**Step 1: Run pytest (algorithm tests)**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 2: Test in Slicer** (manual step)

1. Add `SEEGFellow/` to Slicer's module paths (Edit → Application Settings → Modules → Additional module paths)
2. Restart Slicer
3. Open the SEEGFellow module from the module selector
4. Open Slicer's Python console and run:
   ```python
   tester = slicer.modules.SEEGFellowTest.SEEGFellowTest()
   tester.runTest()
   ```

**Step 3: Fix any issues found during Slicer testing**

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix: address issues found during Slicer integration testing"
```

**Step 5: Merge branch**

```bash
git checkout main
git merge feat/integration-testing
git branch -d feat/integration-testing
```

---

## Summary

| Branch | Tasks | What it delivers |
|--------|-------|-----------------|
| `feat/scaffolding-and-data-model` | 1.1–1.3 | Extension skeleton, dev tooling, data model |
| `feat/metal-segmentation` | 2.1–2.2 | CT metal isolation algorithm + Slicer wrapper |
| `feat/registration` | 3.1 | BRAINSFit rigid registration wrapper |
| `feat/electrode-detection` | 4.1–4.4 | Full automated electrode detection pipeline |
| `feat/trajectory-detector` | 5.1 | Seed-point fallback for missed electrodes |
| `feat/contact-segmenter` | 6.1 | Per-contact cylinder segmentation for 3D viz |
| `feat/gui` | 7.1–7.3 | Wizard UI + widget wiring all steps together |
| `feat/integration-testing` | 8.1–8.2 | Slicer integration tests + bug fixes |
