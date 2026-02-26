from __future__ import annotations

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
    """Wizard-style widget that guides users through the SEEG localization pipeline."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SEEGFellow.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.logic = SEEGFellowLogic()

        # Step 1: Load Data
        self.ui.loadButton.clicked.connect(self._on_load_clicked)

        # Step 2: Rough Alignment
        self.ui.createTransformButton.clicked.connect(self._on_create_transform_clicked)
        self.ui.openTransformsButton.clicked.connect(self._on_open_transforms_clicked)
        self.ui.roughAlignDoneButton.clicked.connect(self._on_rough_align_done_clicked)

        # Step 3: Co-registration
        self.ui.registerButton.clicked.connect(self._on_register_clicked)
        self.ui.acceptRegistrationButton.clicked.connect(
            self._on_accept_registration_clicked
        )
        self.ui.rerunRegistrationButton.clicked.connect(self._on_register_clicked)

        # Step 4: Metal Segmentation
        self.ui.segmentMetalButton.clicked.connect(self._on_segment_metal_clicked)
        self.ui.acceptSegmentationButton.clicked.connect(
            self._on_accept_segmentation_clicked
        )
        self.ui.adjustSegmentationButton.clicked.connect(self._on_segment_metal_clicked)

        # Step 5: Electrode Detection
        self.ui.detectElectrodesButton.clicked.connect(
            self._on_detect_electrodes_clicked
        )
        self.ui.applyNamesButton.clicked.connect(self._on_apply_names_clicked)

        # Step 6: Manual Fallback
        self.ui.placeSeedButton.clicked.connect(self._on_place_seed_clicked)
        self.ui.placeDirectionButton.clicked.connect(self._on_place_direction_clicked)
        self.ui.detectSingleButton.clicked.connect(self._on_detect_single_clicked)

        # Results & Export
        self.ui.createSegmentationButton.clicked.connect(
            self._on_create_segmentation_clicked
        )
        self.ui.exportCsvButton.clicked.connect(self._on_export_csv_clicked)

        # Electrode List
        self.ui.deleteElectrodeButton.clicked.connect(self._on_delete_electrode_clicked)

        # Set up electrode table columns
        self.ui.electrodeTable.setColumnCount(3)
        self.ui.electrodeTable.setHorizontalHeaderLabels(
            ["Contacts", "Spacing (mm)", "Name"]
        )
        self.ui.electrodeTable.horizontalHeader().setStretchLastSection(True)

        # Set up contact table columns
        self.ui.contactTable.setColumnCount(5)
        self.ui.contactTable.setHorizontalHeaderLabels(
            ["Electrode", "Contact", "R", "A", "S"]
        )
        self.ui.contactTable.horizontalHeader().setStretchLastSection(True)

    def cleanup(self):
        self.logic.cleanup()

    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------

    def _on_load_clicked(self):
        t1_path = self.ui.t1PathLineEdit.currentPath
        ct_path = self.ui.ctPathLineEdit.currentPath
        if not t1_path or not ct_path:
            slicer.util.errorDisplay("Please select both T1 and CT files.")
            return
        try:
            slicer.util.showStatusMessage("Loading volumes...")
            self.logic.load_volumes(t1_path, ct_path)
            slicer.util.showStatusMessage("Volumes loaded.")
            self.ui.roughAlignCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load volumes: {e}")

    # -------------------------------------------------------------------------
    # Step 2: Rough Alignment
    # -------------------------------------------------------------------------

    def _on_create_transform_clicked(self):
        try:
            self.logic.create_rough_transform()
            slicer.util.showStatusMessage(
                "Transform created. Adjust in Transforms module."
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to create transform: {e}")

    def _on_open_transforms_clicked(self):
        slicer.util.selectModule("Transforms")

    def _on_rough_align_done_clicked(self):
        self.ui.registrationCollapsibleButton.collapsed = False

    # -------------------------------------------------------------------------
    # Step 3: Co-registration
    # -------------------------------------------------------------------------

    def _on_register_clicked(self):
        try:
            self.ui.registrationProgressBar.setValue(0)
            slicer.util.showStatusMessage("Running BRAINSFit registration...")
            self.logic.run_registration()
            self.ui.registrationProgressBar.setValue(100)
            slicer.util.showStatusMessage("Registration complete.")
        except Exception as e:
            slicer.util.errorDisplay(f"Registration failed: {e}")

    def _on_accept_registration_clicked(self):
        self.ui.metalSegCollapsibleButton.collapsed = False

    # -------------------------------------------------------------------------
    # Step 4: Metal Segmentation
    # -------------------------------------------------------------------------

    def _on_segment_metal_clicked(self):
        threshold = self.ui.thresholdSlider.value
        try:
            slicer.util.showStatusMessage("Segmenting metal...")
            self.logic.run_metal_segmentation(threshold)
            slicer.util.showStatusMessage("Metal segmentation complete.")
        except Exception as e:
            slicer.util.errorDisplay(f"Metal segmentation failed: {e}")

    def _on_accept_segmentation_clicked(self):
        self.ui.detectionCollapsibleButton.collapsed = False

    # -------------------------------------------------------------------------
    # Step 5: Electrode Detection
    # -------------------------------------------------------------------------

    def _on_detect_electrodes_clicked(self):
        try:
            slicer.util.showStatusMessage("Detecting electrodes...")
            self.logic.run_electrode_detection()
            self._populate_electrode_table()
            slicer.util.showStatusMessage(
                f"Detected {len(self.logic.electrodes)} electrode(s)."
            )
            self.ui.electrodeListCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Electrode detection failed: {e}")

    def _populate_electrode_table(self):
        from qt import QLineEdit, QTableWidgetItem

        electrodes = self.logic.electrodes
        self.ui.electrodeTable.setRowCount(len(electrodes))
        for row, electrode in enumerate(electrodes):
            self.ui.electrodeTable.setItem(
                row, 0, QTableWidgetItem(str(electrode.num_contacts))
            )
            self.ui.electrodeTable.setItem(
                row, 1, QTableWidgetItem(f"{electrode.params.contact_spacing:.1f}")
            )
            name_edit = QLineEdit(electrode.name)
            self.ui.electrodeTable.setCellWidget(row, 2, name_edit)

        self._refresh_electrode_list()

    def _on_apply_names_clicked(self):
        from qt import QLineEdit

        for row in range(self.ui.electrodeTable.rowCount()):
            name_widget = self.ui.electrodeTable.cellWidget(row, 2)
            if isinstance(name_widget, QLineEdit):
                name = name_widget.text.strip()
                if name and row < len(self.logic.electrodes):
                    self.logic.electrodes[row].assign_labels(name)

        self.logic.update_fiducials()
        self._refresh_electrode_list()
        self._populate_contact_table()
        self.ui.exportCollapsibleButton.collapsed = False
        slicer.util.showStatusMessage("Names applied.")

    def _populate_contact_table(self):
        from qt import QTableWidgetItem

        rows = []
        for electrode in self.logic.electrodes:
            for contact in electrode.contacts:
                r, a, s = contact.position_ras
                rows.append(
                    (electrode.name, contact.label, f"{r:.2f}", f"{a:.2f}", f"{s:.2f}")
                )

        self.ui.contactTable.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                self.ui.contactTable.setItem(row_idx, col_idx, QTableWidgetItem(value))

    def _refresh_electrode_list(self):
        self.ui.electrodeListWidget.clear()
        for electrode in self.logic.electrodes:
            label = electrode.name if electrode.name else "(unnamed)"
            self.ui.electrodeListWidget.addItem(
                f"{label}  ({electrode.num_contacts} contacts)"
            )

    # -------------------------------------------------------------------------
    # Step 6: Manual Fallback
    # -------------------------------------------------------------------------

    def _on_place_seed_clicked(self):
        self.logic.start_seed_placement()
        slicer.util.showStatusMessage("Click on the deepest contact in the slice view.")

    def _on_place_direction_clicked(self):
        self.logic.start_direction_placement()
        slicer.util.showStatusMessage("Click a second point to define the direction.")

    def _on_detect_single_clicked(self):
        num_contacts = self.ui.numContactsSpinBox.value
        spacing = self.ui.spacingSpinBox.value
        try:
            slicer.util.showStatusMessage("Detecting contacts from seed point...")
            self.logic.run_single_detection(num_contacts, spacing)
            self._populate_electrode_table()
            self._refresh_electrode_list()
            slicer.util.showStatusMessage("Single electrode detection complete.")
        except Exception as e:
            slicer.util.errorDisplay(f"Single electrode detection failed: {e}")

    # -------------------------------------------------------------------------
    # Results & Export
    # -------------------------------------------------------------------------

    def _on_create_segmentation_clicked(self):
        try:
            slicer.util.showStatusMessage("Creating contact segmentation...")
            self.logic.create_segmentation()
            slicer.util.showStatusMessage("Segmentation created.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to create segmentation: {e}")

    def _on_export_csv_clicked(self):
        path = slicer.util.saveFileDialog(
            caption="Export contacts as CSV",
            directory="",
            filter="CSV files (*.csv)",
        )
        if not path:
            return
        try:
            self.logic.export_csv(path)
            slicer.util.showStatusMessage(f"Contacts exported to {path}")
        except Exception as e:
            slicer.util.errorDisplay(f"Export failed: {e}")

    # -------------------------------------------------------------------------
    # Electrode List
    # -------------------------------------------------------------------------

    def _on_delete_electrode_clicked(self):
        row = self.ui.electrodeListWidget.currentRow()
        if row < 0 or row >= len(self.logic.electrodes):
            return
        self.logic.delete_electrode(row)
        self._populate_electrode_table()
        self._populate_contact_table()
        self._refresh_electrode_list()
        slicer.util.showStatusMessage("Electrode deleted.")


class SEEGFellowLogic(ScriptedLoadableModuleLogic):
    """Implements the SEEG localization pipeline, independent of the GUI.

    Example (in Slicer Python console)::

        logic = SEEGFellowLogic()
        logic.load_volumes("/path/T1.nii.gz", "/path/CT.nii.gz")
        logic.run_metal_segmentation(threshold=2500)
        logic.run_electrode_detection()
    """

    def __init__(self):
        super().__init__()
        self._t1_node = None
        self._ct_node = None
        self._rough_transform_node = None
        self._registration_transform_node = None
        self._metal_labelmap = None
        self.electrodes: list = []  # list[Electrode]
        self._seed_node = None
        self._direction_node = None

    def cleanup(self):
        """Remove temporary markup nodes."""
        for node in [self._seed_node, self._direction_node]:
            if node is not None:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
        self._seed_node = None
        self._direction_node = None

    # -------------------------------------------------------------------------
    # Step 1: Load volumes
    # -------------------------------------------------------------------------

    def load_volumes(self, t1_path: str, ct_path: str) -> None:
        """Load T1 and CT volumes from disk into the Slicer scene.

        Example::

            logic.load_volumes("/data/T1.nii.gz", "/data/CT.nii.gz")
        """
        self._t1_node = slicer.util.loadVolume(t1_path)
        self._ct_node = slicer.util.loadVolume(ct_path)

    # -------------------------------------------------------------------------
    # Step 2: Rough transform
    # -------------------------------------------------------------------------

    def create_rough_transform(self):
        """Create a linear transform and apply it to the CT for manual rough alignment.

        Example::

            logic.create_rough_transform()
        """
        from SEEGFellowLib.registration import CTtoT1Registration

        reg = CTtoT1Registration()
        self._rough_transform_node = reg.create_rough_transform(self._ct_node)

    # -------------------------------------------------------------------------
    # Step 3: Registration
    # -------------------------------------------------------------------------

    def run_registration(self):
        """Run BRAINSFit rigid registration (CT → T1).

        Example::

            logic.run_registration()
        """
        from SEEGFellowLib.registration import CTtoT1Registration

        reg = CTtoT1Registration()
        self._registration_transform_node = reg.run(
            self._ct_node,
            self._t1_node,
            initial_transform=self._rough_transform_node,
        )

    # -------------------------------------------------------------------------
    # Step 4: Metal segmentation
    # -------------------------------------------------------------------------

    def run_metal_segmentation(self, threshold: float = 2500) -> None:
        """Threshold and clean up CT to isolate metal voxels.

        Example::

            logic.run_metal_segmentation(threshold=2500)
        """
        from SEEGFellowLib.metal_segmenter import MetalSegmenter

        segmenter = MetalSegmenter()
        # Remove previous labelmap if it exists
        if self._metal_labelmap is not None:
            slicer.mrmlScene.RemoveNode(self._metal_labelmap)
        self._metal_labelmap = segmenter.segment(self._ct_node, threshold=threshold)

    # -------------------------------------------------------------------------
    # Step 5: Electrode detection
    # -------------------------------------------------------------------------

    def run_electrode_detection(self) -> None:
        """Run full automated electrode detection pipeline.

        Example::

            logic.run_electrode_detection()
        """
        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        detector = ElectrodeDetector()
        self.electrodes = detector.detect_all(self._metal_labelmap, self._ct_node)
        self._create_fiducials_for_electrodes()

    def _create_fiducials_for_electrodes(self) -> None:
        """Create a markups fiducial node for each electrode's contacts."""
        for electrode in self.electrodes:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode",
                electrode.name if electrode.name else "Electrode",
            )
            for contact in electrode.contacts:
                r, a, s = contact.position_ras
                node.AddControlPoint(r, a, s, contact.label)
            electrode.markups_node_id = node.GetID()

    def update_fiducials(self) -> None:
        """Update fiducial node names and labels after names are assigned."""
        for electrode in self.electrodes:
            node = slicer.mrmlScene.GetNodeByID(electrode.markups_node_id)
            if node is None:
                # Node doesn't exist yet; create it
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", electrode.name
                )
                for contact in electrode.contacts:
                    r, a, s = contact.position_ras
                    node.AddControlPoint(r, a, s, contact.label)
                electrode.markups_node_id = node.GetID()
            else:
                node.SetName(electrode.name)
                # Update control point labels
                for i, contact in enumerate(electrode.contacts):
                    if i < node.GetNumberOfControlPoints():
                        node.SetNthControlPointLabel(i, contact.label)

    # -------------------------------------------------------------------------
    # Step 6: Manual fallback (seed-point)
    # -------------------------------------------------------------------------

    def start_seed_placement(self) -> None:
        """Enter markup placement mode to let the user click the seed contact."""
        if self._seed_node is None:
            self._seed_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "SeedPoint"
            )
            self._seed_node.SetMaximumNumberOfControlPoints(1)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self._seed_node.GetID())
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def start_direction_placement(self) -> None:
        """Enter markup placement mode to let the user click a direction hint."""
        if self._direction_node is None:
            self._direction_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "DirectionHint"
            )
            self._direction_node.SetMaximumNumberOfControlPoints(1)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self._direction_node.GetID())
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def run_single_detection(self, num_contacts: int, spacing: float) -> None:
        """Detect contacts along a single electrode from a seed point.

        Example::

            logic.run_single_detection(num_contacts=8, spacing=3.5)
        """
        import numpy as np
        from SEEGFellowLib.electrode_model import Electrode, ElectrodeParams
        from SEEGFellowLib.trajectory_detector import IntensityProfileDetector

        if self._seed_node is None or self._seed_node.GetNumberOfControlPoints() == 0:
            raise RuntimeError("No seed point placed. Click 'Place Seed Point' first.")

        seed_pos = [0.0, 0.0, 0.0]
        self._seed_node.GetNthControlPointPosition(0, seed_pos)
        seed_ras: tuple[float, float, float] = (seed_pos[0], seed_pos[1], seed_pos[2])

        direction_hint: tuple[float, float, float] | None = None
        if (
            self._direction_node is not None
            and self._direction_node.GetNumberOfControlPoints() > 0
        ):
            hint_pos = [0.0, 0.0, 0.0]
            self._direction_node.GetNthControlPointPosition(0, hint_pos)
            diff = np.array(hint_pos) - np.array(seed_pos)
            direction_hint = (float(diff[0]), float(diff[1]), float(diff[2]))

        params = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=spacing,
            contact_diameter=0.8,
        )

        detector = IntensityProfileDetector()
        contacts = detector.detect(
            seed_ras=seed_ras,
            ct_volume_node=self._ct_node,
            num_contacts=num_contacts,
            params=params,
            direction_hint=direction_hint,
        )

        electrode = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=(0.0, 0.0, 0.0),
        )
        self.electrodes.append(electrode)

        # Create fiducials for the new electrode
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "ManualElectrode"
        )
        for contact in contacts:
            r, a, s = contact.position_ras
            node.AddControlPoint(r, a, s, contact.label)
        electrode.markups_node_id = node.GetID()

    # -------------------------------------------------------------------------
    # Results & Export
    # -------------------------------------------------------------------------

    def create_segmentation(self) -> None:
        """Create a per-contact cylindrical segmentation for 3D visualization.

        Example::

            logic.create_segmentation()
        """
        from SEEGFellowLib.contact_segmenter import ContactSegmenter

        segmenter = ContactSegmenter()
        segmenter.create_segmentation(self.electrodes, self._ct_node)

    def export_csv(self, path: str) -> None:
        """Export all contact positions to a CSV file.

        Example::

            logic.export_csv("/output/contacts.csv")
        """
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Electrode", "Contact", "R", "A", "S"])
            for electrode in self.electrodes:
                for contact in electrode.contacts:
                    r, a, s = contact.position_ras
                    writer.writerow([electrode.name, contact.label, r, a, s])

    def delete_electrode(self, index: int) -> None:
        """Remove an electrode and its fiducials from the session.

        Example::

            logic.delete_electrode(0)
        """
        if index < 0 or index >= len(self.electrodes):
            return
        electrode = self.electrodes[index]
        node = slicer.mrmlScene.GetNodeByID(electrode.markups_node_id)
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        self.electrodes.pop(index)


class SEEGFellowTest(ScriptedLoadableModuleTest):
    def runTest(self):
        pass
