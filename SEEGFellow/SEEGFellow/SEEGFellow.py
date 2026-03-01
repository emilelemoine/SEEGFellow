from __future__ import annotations

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from SEEGFellowLib.hemisphere_labels import (
    LEFT_HEMISPHERE_LABELS as _LEFT_HEMISPHERE_LABELS,
    RIGHT_HEMISPHERE_LABELS as _RIGHT_HEMISPHERE_LABELS,
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


def _strip_hemisphere(region: str) -> str:
    """Remove leading 'Left ' or 'Right ' from an anatomical region name."""
    for prefix in ("Left ", "Right "):
        if region.startswith(prefix):
            return region[len(prefix) :]
    return region


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

        # Step 2: Co-registration
        self.ui.createTransformButton.clicked.connect(self._on_create_transform_clicked)
        self.ui.registerButton.clicked.connect(self._on_register_clicked)

        # Step 3a: Intracranial Mask
        self._setup_synthseg_ui()
        self.ui.computeHeadMaskButton.clicked.connect(
            self._on_compute_head_mask_clicked
        )
        self.ui.editHeadMaskButton.clicked.connect(self._on_edit_head_mask_clicked)

        # Step 3b: Metal Threshold
        self.ui.thresholdSlider.valueChanged.connect(self._on_threshold_changed)
        self.ui.applyMetalThresholdButton.clicked.connect(
            self._on_apply_metal_threshold_clicked
        )
        self.ui.editMetalMaskButton.clicked.connect(self._on_edit_metal_mask_clicked)

        # Step 3c: Contact Detection
        self.ui.detectElectrodesButton.clicked.connect(
            self._on_detect_electrodes_clicked
        )

        # Step 4: Rename Electrodes
        self.ui.applyNamesButton.clicked.connect(self._on_apply_names_clicked)

        # Step 5: Label Contacts
        self.ui.labelContactsButton.clicked.connect(self._on_label_contacts_clicked)
        self._left_anatomy_electrodes: list = []
        self._right_anatomy_electrodes: list = []
        self.ui.leftAnatomyTable.cellClicked.connect(
            lambda row, col: self._on_anatomy_table_cell_clicked(
                row, col, self._left_anatomy_electrodes
            )
        )
        self.ui.rightAnatomyTable.cellClicked.connect(
            lambda row, col: self._on_anatomy_table_cell_clicked(
                row, col, self._right_anatomy_electrodes
            )
        )
        self.ui.leftAnatomyCopyButton.clicked.connect(
            lambda: self._on_anatomy_copy_html(self._left_anatomy_electrodes)
        )
        self.ui.rightAnatomyCopyButton.clicked.connect(
            lambda: self._on_anatomy_copy_html(self._right_anatomy_electrodes)
        )
        self.ui.leftAnatomyExportButton.clicked.connect(
            lambda: self._on_anatomy_export_tsv(self._left_anatomy_electrodes)
        )
        self.ui.rightAnatomyExportButton.clicked.connect(
            lambda: self._on_anatomy_export_tsv(self._right_anatomy_electrodes)
        )

        # Results & Export
        self.ui.exportCsvButton.clicked.connect(self._on_export_csv_clicked)

        # Electrode List
        self.ui.deleteElectrodeButton.clicked.connect(self._on_delete_electrode_clicked)

        # Maps table row → index in self.logic.electrodes for each hemisphere table
        self._left_electrode_indices: list[int] = []
        self._right_electrode_indices: list[int] = []

        # Set up left/right electrode table columns
        for table in (self.ui.leftElectrodeTable, self.ui.rightElectrodeTable):
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Label", "Contacts", "Note"])
            table.horizontalHeader().setStretchLastSection(True)
            # Ensure at least 16 rows are visible even when the table is empty
            table.setMinimumHeight(530)
        self.ui.leftElectrodeTable.cellClicked.connect(
            lambda row, col: self._on_electrode_table_row_clicked(
                row, col, self.ui.leftElectrodeTable, self._left_electrode_indices
            )
        )
        self.ui.rightElectrodeTable.cellClicked.connect(
            lambda row, col: self._on_electrode_table_row_clicked(
                row, col, self.ui.rightElectrodeTable, self._right_electrode_indices
            )
        )

        # Set up contact table columns
        self.ui.contactTable.setColumnCount(6)
        self.ui.contactTable.setHorizontalHeaderLabels(
            ["Electrode", "Contact", "R", "A", "S", "Region"]
        )
        self.ui.contactTable.horizontalHeader().setStretchLastSection(True)

        # Auto-restore from saved scene
        self._try_restore_session()

    def _setup_synthseg_ui(self):
        """Check FreeSurfer availability and configure status label."""
        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        strategy = SynthSegBrainMask()
        if strategy.is_available():
            self.ui.freesurferStatusLabel.setText("Found")
            self.ui.freesurferPathLabel.visible = False
            self.ui.freesurferPathLineEdit.visible = False
        else:
            self.ui.freesurferStatusLabel.setText("Not found — set path below")
            self.ui.freesurferPathLabel.visible = True
            self.ui.freesurferPathLineEdit.visible = True
            # Check Slicer settings for a saved path
            saved_path = slicer.app.settings().value("SEEGFellow/FreeSurferHome", "")
            if saved_path:
                self.ui.freesurferPathLineEdit.currentPath = saved_path

    def _try_restore_session(self):
        """Attempt to reconnect to nodes from a saved Slicer scene."""
        if not self.logic.try_restore_from_scene():
            return

        # Determine furthest-reached step and uncollapse that panel
        has_brain = self.logic._head_mask is not None
        has_metal = self.logic._metal_mask is not None
        has_parcellation = self.logic._parcellation is not None
        has_electrodes = len(self.logic.electrodes) > 0

        if has_parcellation and has_electrodes:
            self.ui.labelContactsCollapsibleButton.collapsed = False
        elif has_metal:
            self.ui.contactDetectionCollapsibleButton.collapsed = False
        elif has_brain:
            self.ui.metalThresholdCollapsibleButton.collapsed = False
        else:
            self.ui.intracranialMaskCollapsibleButton.collapsed = False

        slicer.util.showStatusMessage("Restored session from scene.")

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
            self.ui.coregistrationCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load volumes: {e}")

    # -------------------------------------------------------------------------
    # Step 2: Co-registration
    # -------------------------------------------------------------------------

    def _on_create_transform_clicked(self):
        self._ensure_session_restored()
        try:
            self.logic.create_rough_transform()
            self._setup_registration_view()
            slicer.util.selectModule("Transforms")
            slicer.util.showStatusMessage(
                "Transform created. Adjust in Transforms module."
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to create transform: {e}")

    def _setup_registration_view(self) -> None:
        """Show T1 (foreground, 50%) over CT in all slice views for registration."""
        t1_node = self.logic._t1_node
        ct_node = self.logic._ct_node
        if t1_node is None or ct_node is None:
            return
        layout_manager = slicer.app.layoutManager()
        for name in layout_manager.sliceViewNames():
            composite = (
                layout_manager.sliceWidget(name).sliceLogic().GetSliceCompositeNode()
            )
            composite.SetBackgroundVolumeID(ct_node.GetID())
            composite.SetForegroundVolumeID(t1_node.GetID())
            composite.SetForegroundOpacity(0.5)

    def _on_register_clicked(self):
        self._ensure_session_restored()
        try:
            slicer.util.showStatusMessage("Running BRAINSFit registration...")
            self.logic.run_registration()
            slicer.util.showStatusMessage("Registration complete.")
            self.ui.intracranialMaskCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Registration failed: {e}")

    # -------------------------------------------------------------------------
    # Step 4a: Intracranial Mask
    # -------------------------------------------------------------------------

    def _ensure_session_restored(self) -> None:
        """Lazily restore scene nodes if the scene was loaded after module init."""
        if self.logic._t1_node is None:
            self.logic.try_restore_from_scene()

    def _on_compute_head_mask_clicked(self):
        self._ensure_session_restored()

        # Resolve FreeSurfer path if set via the UI browse widget
        fs_path = self.ui.freesurferPathLineEdit.currentPath
        if fs_path:
            import os

            os.environ["FREESURFER_HOME"] = fs_path
            slicer.app.settings().setValue("SEEGFellow/FreeSurferHome", fs_path)

        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        robust = self.ui.synthSegModeComboBox.currentIndex == 0
        threads = self.ui.synthSegThreadsSpinBox.value
        strategy = SynthSegBrainMask(robust=robust, threads=threads)

        if not strategy.is_available():
            slicer.util.errorDisplay(
                "FreeSurfer not found. Set FREESURFER_HOME or browse to the install directory."
            )
            return

        output_dir = self.ui.synthSegOutputDirLineEdit.currentPath or None

        try:
            slicer.util.showStatusMessage("Running SynthSeg brain segmentation...")
            self.logic.run_intracranial_mask(strategy=strategy, output_dir=output_dir)
            slicer.util.showStatusMessage("Brain segmentation complete.")
            self.ui.metalThresholdCollapsibleButton.collapsed = False
            # Update status label
            self.ui.freesurferStatusLabel.setText("Found")
            self.ui.freesurferPathLabel.visible = False
            self.ui.freesurferPathLineEdit.visible = False
        except Exception as e:
            slicer.util.errorDisplay(f"SynthSeg failed: {e}")

    def _on_edit_head_mask_clicked(self):
        if self.logic._segmentation_node is None:
            slicer.util.errorDisplay("Run 'Compute Brain Mask' first.")
            return
        slicer.util.selectModule("SegmentEditor")
        editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self()
        # .editor is the underlying qMRMLSegmentEditorWidget
        editor_widget.editor.setSegmentationNode(self.logic._segmentation_node)
        # Use CT as background so the user can see electrodes while editing
        editor_widget.editor.setSourceVolumeNode(self.logic._ct_node)

    # -------------------------------------------------------------------------
    # Step 3b: Metal Threshold
    # -------------------------------------------------------------------------

    def _on_threshold_changed(self, value: float) -> None:
        ct_node = self.logic._ct_node
        if ct_node is None:
            self.ui.voxelCountLabel.setText("—")
            return
        import numpy as np
        from slicer.util import arrayFromVolume

        ct_array = arrayFromVolume(ct_node)
        count = int(np.sum(ct_array >= value))
        self.ui.voxelCountLabel.setText(f"{count:,} voxels above threshold")

    def _on_apply_metal_threshold_clicked(self):
        self._ensure_session_restored()
        threshold = self.ui.thresholdSlider.value
        try:
            slicer.util.showStatusMessage("Applying metal threshold...")
            self.logic.run_metal_threshold(threshold)
            slicer.util.showStatusMessage("Metal threshold applied.")
            self.ui.contactDetectionCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to apply threshold: {e}")

    def _on_edit_metal_mask_clicked(self):
        if self.logic._segmentation_node is None:
            slicer.util.errorDisplay("Run 'Apply Threshold' first.")
            return
        slicer.util.selectModule("SegmentEditor")
        editor_widget = slicer.modules.segmenteditor.widgetRepresentation().self()
        editor_widget.editor.setSegmentationNode(self.logic._segmentation_node)
        editor_widget.editor.setSourceVolumeNode(self.logic._ct_node)

    # -------------------------------------------------------------------------
    # Step 3c: Contact Detection
    # -------------------------------------------------------------------------

    def _on_detect_electrodes_clicked(self):
        self._ensure_session_restored()
        sigma = self.ui.sigmaSlider.value
        expected_spacing = self.ui.expectedSpacingSpinBox.value
        min_contacts = self.ui.minContactsSpinBox.value
        max_component_voxels = self.ui.maxComponentVoxelsSpinBox.value
        # Slider stores percentage (30–90); convert to factor (0.30–0.90)
        spacing_cutoff_factor = self.ui.spacingCutoffSlider.value / 100.0
        distance_tolerance = self.ui.distanceToleranceSpinBox.value
        max_iterations = self.ui.maxIterationsSpinBox.value
        try:
            slicer.util.showStatusMessage("Detecting electrodes...")
            self.logic.run_electrode_detection(
                sigma=sigma,
                expected_spacing=expected_spacing,
                min_contacts=min_contacts,
                max_component_voxels=max_component_voxels,
                spacing_cutoff_factor=spacing_cutoff_factor,
                distance_tolerance=distance_tolerance,
                max_iterations=max_iterations,
            )
            self._populate_electrode_table()
            # Hide the segmentation overlay so contacts are visible
            if self.logic._segmentation_node is not None:
                self.logic._segmentation_node.GetDisplayNode().SetVisibility(False)
            slicer.util.showStatusMessage(
                f"Detected {len(self.logic.electrodes)} electrode(s)."
            )
            self.ui.renameElectrodesCollapsibleButton.collapsed = False
            self.ui.electrodeListCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Electrode detection failed: {e}")

    def _populate_electrode_table(self):
        from qt import QLineEdit, QTableWidgetItem

        electrodes = self.logic.electrodes

        # Split electrodes into left (R < 0) and right (R >= 0) by first contact RAS X
        left_indices: list[int] = []
        right_indices: list[int] = []
        for idx, electrode in enumerate(electrodes):
            x = electrode.contacts[-1].position_ras[0] if electrode.contacts else 0.0
            if x < 0:
                left_indices.append(idx)
            else:
                right_indices.append(idx)

        self._left_electrode_indices = left_indices
        self._right_electrode_indices = right_indices

        def _fill_table(table, indices):
            table.setRowCount(len(indices))
            for row, idx in enumerate(indices):
                electrode = electrodes[idx]
                name_edit = QLineEdit(electrode.name)
                table.setCellWidget(row, 0, name_edit)
                table.setItem(row, 1, QTableWidgetItem(str(electrode.num_contacts)))
                table.setItem(row, 2, QTableWidgetItem(""))

        _fill_table(self.ui.leftElectrodeTable, left_indices)
        _fill_table(self.ui.rightElectrodeTable, right_indices)

        self._refresh_electrode_list()

    def _on_electrode_table_row_clicked(
        self, row: int, _column: int, table, indices: list[int]
    ) -> None:
        """Highlight the selected electrode in red and jump slice views to it."""
        if row < 0 or row >= len(indices):
            return

        selected_idx = indices[row]
        colors = SEEGFellowLogic.ELECTRODE_COLORS
        for idx, electrode in enumerate(self.logic.electrodes):
            node = slicer.mrmlScene.GetNodeByID(electrode.markups_node_id)
            if node is None:
                continue
            display = node.GetDisplayNode()
            if idx == selected_idx:
                display.SetSelectedColor(1.0, 0.0, 0.0)
                display.SetColor(1.0, 0.0, 0.0)
                display.SetOpacity(1.0)
            else:
                color = colors[idx % len(colors)]
                display.SetSelectedColor(*color)
                display.SetColor(*color)
                display.SetOpacity(0.7)

        # Jump slice views to the first contact of the selected electrode
        electrode = self.logic.electrodes[selected_idx]
        if electrode.contacts:
            r, a, s = electrode.contacts[0].position_ras
            slicer.modules.markups.logic().JumpSlicesToLocation(r, a, s, True)

    def _on_apply_names_clicked(self):
        from qt import QLineEdit

        for table, indices in (
            (self.ui.leftElectrodeTable, self._left_electrode_indices),
            (self.ui.rightElectrodeTable, self._right_electrode_indices),
        ):
            for row in range(table.rowCount):
                name_widget = table.cellWidget(row, 0)
                if isinstance(name_widget, QLineEdit):
                    name = name_widget.text.strip()
                    if name and row < len(indices):
                        self.logic.electrodes[indices[row]].assign_labels(name)

        self.logic.update_fiducials()
        self._restore_electrode_colors()
        self._refresh_electrode_list()
        self._populate_contact_table()
        self.ui.labelContactsCollapsibleButton.collapsed = False
        slicer.util.showStatusMessage("Names applied.")

    def _on_label_contacts_clicked(self):
        self._ensure_session_restored()
        try:
            slicer.util.showStatusMessage("Labeling contacts...")
            self.logic.run_contact_labeling()
            self._populate_anatomy_table()
            slicer.util.showStatusMessage("Contact labeling complete.")
            self.ui.exportCollapsibleButton.collapsed = False
        except Exception as e:
            slicer.util.errorDisplay(f"Contact labeling failed: {e}")

    def _populate_anatomy_table(self):
        from qt import QTableWidgetItem

        electrodes = self.logic.electrodes
        if not electrodes:
            return

        # Split by hemisphere: R >= 0 is right, R < 0 is left
        left_electrodes = sorted(
            [
                e
                for e in electrodes
                if e.contacts and e.contacts[-1].position_ras[0] < 0
            ],
            key=lambda e: e.name,
        )
        right_electrodes = sorted(
            [
                e
                for e in electrodes
                if e.contacts and e.contacts[-1].position_ras[0] >= 0
            ],
            key=lambda e: e.name,
        )

        def _fill(table, hemisphere_electrodes):
            if not hemisphere_electrodes:
                table.setRowCount(0)
                table.setColumnCount(0)
                return
            max_contacts = max(e.num_contacts for e in hemisphere_electrodes)
            table.setRowCount(len(hemisphere_electrodes))
            table.setColumnCount(max_contacts)
            table.setHorizontalHeaderLabels([str(i + 1) for i in range(max_contacts)])
            table.setVerticalHeaderLabels([e.name for e in hemisphere_electrodes])
            for row, electrode in enumerate(hemisphere_electrodes):
                for col, contact in enumerate(electrode.contacts):
                    display_region = _strip_hemisphere(contact.region)
                    item = QTableWidgetItem(display_region)
                    item.setToolTip(contact.region)
                    table.setItem(row, col, item)
            # Narrow fixed column width
            for col in range(max_contacts):
                table.setColumnWidth(col, 90)
            # Show at least 15 rows (~30px/row + 30px header)
            table.setMinimumHeight(480)

        self._right_anatomy_electrodes = right_electrodes
        self._left_anatomy_electrodes = left_electrodes
        _fill(self.ui.rightAnatomyTable, right_electrodes)
        _fill(self.ui.leftAnatomyTable, left_electrodes)

    def _on_anatomy_table_cell_clicked(
        self, row: int, col: int, electrodes_list: list
    ) -> None:
        """Highlight the clicked contact's electrode in red and jump slice views to it."""
        if row < 0 or row >= len(electrodes_list):
            return
        electrode = electrodes_list[row]
        if col < 0 or col >= len(electrode.contacts):
            return

        colors = SEEGFellowLogic.ELECTRODE_COLORS
        for idx, e in enumerate(self.logic.electrodes):
            node = slicer.mrmlScene.GetNodeByID(e.markups_node_id)
            if node is None:
                continue
            display = node.GetDisplayNode()
            if e is electrode:
                display.SetSelectedColor(1.0, 0.0, 0.0)
                display.SetColor(1.0, 0.0, 0.0)
                display.SetOpacity(1.0)
            else:
                color = colors[idx % len(colors)]
                display.SetSelectedColor(*color)
                display.SetColor(*color)
                display.SetOpacity(0.7)

        r, a, s = electrode.contacts[col].position_ras
        slicer.modules.markups.logic().JumpSlicesToLocation(r, a, s, True)

    def _anatomy_to_tsv(self, electrodes_list: list) -> str:
        """Build a TSV string from the anatomy electrode list (hemisphere-prefix stripped)."""
        if not electrodes_list:
            return ""
        max_contacts = max(len(e.contacts) for e in electrodes_list)
        header = "\t".join(["Electrode"] + [str(i + 1) for i in range(max_contacts)])
        rows = [header]
        for electrode in electrodes_list:
            cells = [electrode.name] + [
                _strip_hemisphere(c.region) for c in electrode.contacts
            ]
            rows.append("\t".join(cells))
        return "\n".join(rows)

    def _on_anatomy_copy_html(self, electrodes_list: list) -> None:
        """Copy the anatomy table as an HTML table to the clipboard."""
        import qt

        if not electrodes_list:
            return
        max_contacts = max(len(e.contacts) for e in electrodes_list)
        header_cells = "".join(f"<th>{i + 1}</th>" for i in range(max_contacts))
        body_rows = ""
        for electrode in electrodes_list:
            cells = "".join(
                f"<td>{_strip_hemisphere(c.region)}</td>" for c in electrode.contacts
            )
            cells += "<td></td>" * (max_contacts - len(electrode.contacts))
            body_rows += f"<tr><th>{electrode.name}</th>{cells}</tr>"
        html = (
            f"<table><thead><tr><th>Electrode</th>{header_cells}</tr></thead>"
            f"<tbody>{body_rows}</tbody></table>"
        )
        mime = qt.QMimeData()
        mime.setHtml(html)
        mime.setText(self._anatomy_to_tsv(electrodes_list))
        qt.QApplication.clipboard().setMimeData(mime)
        slicer.util.showStatusMessage("Anatomy table copied as HTML.")

    def _on_anatomy_export_tsv(self, electrodes_list: list) -> None:
        """Export the anatomy table as a TSV file."""
        if not electrodes_list:
            return
        path = slicer.util.saveFileDialog(
            caption="Export anatomy as TSV",
            directory="",
            filter="TSV files (*.tsv)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._anatomy_to_tsv(electrodes_list))
            slicer.util.showStatusMessage(f"Anatomy exported to {path}")
        except Exception as e:
            slicer.util.errorDisplay(f"Export failed: {e}")

    def _restore_electrode_colors(self) -> None:
        """Reset all electrode fiducial colors to their assigned palette colors."""
        colors = SEEGFellowLogic.ELECTRODE_COLORS
        for idx, electrode in enumerate(self.logic.electrodes):
            node = slicer.mrmlScene.GetNodeByID(electrode.markups_node_id)
            if node is None:
                continue
            display = node.GetDisplayNode()
            color = colors[idx % len(colors)]
            display.SetSelectedColor(*color)
            display.SetColor(*color)

    def _populate_contact_table(self):
        from qt import QTableWidgetItem

        self.ui.contactTable.setColumnCount(6)
        self.ui.contactTable.setHorizontalHeaderLabels(
            ["Electrode", "Contact", "R", "A", "S", "Region"]
        )
        self.ui.contactTable.horizontalHeader().setStretchLastSection(True)

        rows = []
        for electrode in self.logic.electrodes:
            for contact in electrode.contacts:
                r, a, s = contact.position_ras
                rows.append(
                    (
                        electrode.name,
                        contact.label,
                        f"{r:.2f}",
                        f"{a:.2f}",
                        f"{s:.2f}",
                        contact.region,
                    )
                )

        self.ui.contactTable.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                self.ui.contactTable.setItem(row_idx, col_idx, QTableWidgetItem(value))

    def _refresh_electrode_list(self):
        self.ui.electrodeListWidget.clear()
        for electrode in self.logic.electrodes:
            label = electrode.name if electrode.name else "(unnamed)"
            spacing = electrode.params.contact_spacing
            self.ui.electrodeListWidget.addItem(
                f"{label}  ({electrode.num_contacts} contacts, {spacing:.1f} mm)"
            )

    # -------------------------------------------------------------------------
    # Results & Export
    # -------------------------------------------------------------------------

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
        logic.run_electrode_detection(sigma=1.2, expected_spacing=3.5)
    """

    def __init__(self):
        super().__init__()
        self._t1_node = None
        self._ct_node = None
        self._rough_transform_node = None
        self._registration_transform_node = None
        self.electrodes: list = []  # list[Electrode]
        self._raw_contacts_node_id: str | None = None
        self._segmentation_node = None
        self._head_mask = None
        self._metal_mask = None
        self._parcellation = None
        self._parcellation_affine = None

    def cleanup(self):
        pass

    # -------------------------------------------------------------------------
    # Session restore
    # -------------------------------------------------------------------------

    def try_restore_from_scene(self) -> bool:
        """Reconnect to existing scene nodes from a saved Slicer scene.

        Scans for nodes by name convention:
        - Transform: "CT_to_T1_Registration"
        - CT: scalar volume with that transform as parent
        - T1: the other scalar volume
        - Segmentation: "SEEGFellow Segmentation" (optional)

        Returns True if CT + T1 + registration transform were found.
        """
        # --- Find registration transform ---
        transform = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLLinearTransformNode", "CT_to_T1_Registration"
        )
        if transform is None:
            return False

        # --- Find CT (volume under the registration transform) ---
        scene = slicer.mrmlScene
        ct_node = None
        other_volumes = []
        n = scene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode")
        for i in range(n):
            vol = scene.GetNthNodeByClass(i, "vtkMRMLScalarVolumeNode")
            if vol.GetParentTransformNode() == transform:
                if ct_node is not None:
                    return False  # ambiguous: 2+ volumes under transform
                ct_node = vol
            else:
                other_volumes.append(vol)

        if ct_node is None or len(other_volumes) != 1:
            return False

        t1_node = other_volumes[0]

        # --- Assign core nodes ---
        self._registration_transform_node = transform
        self._ct_node = ct_node
        self._t1_node = t1_node

        # --- Optionally restore segmentation and masks ---
        seg_node = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLSegmentationNode", "SEEGFellow Segmentation"
        )
        import numpy as np

        if seg_node is not None:
            self._segmentation_node = seg_node
            seg = seg_node.GetSegmentation()

            brain_id = seg.GetSegmentIdBySegmentName("Brain")
            if brain_id:
                brain_array = slicer.util.arrayFromSegmentBinaryLabelmap(
                    seg_node, brain_id, ct_node
                )
                self._head_mask = (brain_array > 0).astype(np.uint8)

            metal_id = seg.GetSegmentIdBySegmentName("Metal")
            if metal_id:
                metal_array = slicer.util.arrayFromSegmentBinaryLabelmap(
                    seg_node, metal_id, ct_node
                )
                self._metal_mask = (metal_array > 0).astype(np.uint8)

        # Restore parcellation if saved
        parc_node = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_SynthSeg_Parcellation"
        )
        if parc_node is not None:
            import vtk

            self._parcellation = np.array(
                slicer.util.arrayFromVolume(parc_node), dtype=np.int32
            )
            mat = vtk.vtkMatrix4x4()
            parc_node.GetIJKToRASMatrix(mat)
            self._parcellation_affine = np.array(
                [[mat.GetElement(r, c) for c in range(4)] for r in range(4)]
            )

        return True

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
    # Step 4: Electrode detection
    # -------------------------------------------------------------------------

    def run_intracranial_mask(
        self, strategy=None, output_dir: str | None = None
    ) -> None:
        """Compute brain parenchyma mask from the T1 MRI and display it.

        The mask is computed in MRI space using the given strategy, then
        resampled into CT space so it can be used to classify electrode
        contacts.  The T1 must already be loaded (Step 1) and registered
        to the CT (Step 3).

        Args:
            strategy: A BrainMaskStrategy instance.  Defaults to
                SynthSegBrainMask if not provided.
            output_dir: If set, SynthSeg output is saved here for reuse.

        Example::

            logic.run_intracranial_mask()
        """
        import numpy as np
        import vtk
        from slicer.util import arrayFromVolume, updateVolumeFromArray

        if strategy is None:
            from SEEGFellowLib.brain_mask import SynthSegBrainMask

            strategy = SynthSegBrainMask()

        if self._t1_node is None:
            raise RuntimeError("T1 MRI not loaded. Complete Step 1 first.")

        # --- Compute mask in MRI voxel space ---
        t1_array = arrayFromVolume(self._t1_node)

        # Extract the 4x4 IJK-to-RAS affine (needed by brain extraction)
        ijkToRAS = vtk.vtkMatrix4x4()
        self._t1_node.GetIJKToRASMatrix(ijkToRAS)
        affine = np.array(
            [[ijkToRAS.GetElement(r, c) for c in range(4)] for r in range(4)]
        )

        brain_mask_t1 = strategy.compute(t1_array, affine, output_dir=output_dir)

        # Store parcellation for downstream contact labeling
        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        if (
            isinstance(strategy, SynthSegBrainMask)
            and strategy.parcellation is not None
        ):
            self._parcellation = strategy.parcellation
            self._parcellation_affine = strategy.parcellation_affine

        # Save parcellation as a label map node for scene persistence
        if self._parcellation is not None:
            parc_node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_SynthSeg_Parcellation"
            )
            if parc_node is None:
                parc_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_SynthSeg_Parcellation"
                )
            # Set geometry from parcellation affine
            mat = vtk.vtkMatrix4x4()
            for r_idx in range(4):
                for c_idx in range(4):
                    mat.SetElement(
                        r_idx, c_idx, float(self._parcellation_affine[r_idx, c_idx])
                    )
            parc_node.SetIJKToRASMatrix(mat)
            # Store the parcellation (transpose from Slicer K,J,I back to NIfTI I,J,K
            # for the label map node, then Slicer handles it internally)
            updateVolumeFromArray(parc_node, self._parcellation)
            parc_node.SetHideFromEditors(True)

        print(
            f"[SEEGFellow] Brain mask voxel count in MRI space: "
            f"{np.sum(brain_mask_t1 > 0)}"
        )

        # --- Create temporary nodes for resampling ---
        brain_label_node = None
        brain_label_ct = None
        try:
            # --- Create a temporary labelmap in MRI space ---
            # SynthSeg resamples to 1mm isotropic, so the mask may have different
            # dimensions/spacing than the T1.  Use the strategy's output affine when
            # available; fall back to the T1 IJKToRAS for strategies that return
            # masks in the original T1 voxel grid.
            brain_label_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_BrainMask_MRI"
            )
            mask_affine_np = getattr(strategy, "parcellation_affine", None)
            if mask_affine_np is not None:
                mask_mat = vtk.vtkMatrix4x4()
                for r_idx in range(4):
                    for c_idx in range(4):
                        mask_mat.SetElement(
                            r_idx, c_idx, float(mask_affine_np[r_idx, c_idx])
                        )
                brain_label_node.SetIJKToRASMatrix(mask_mat)
            else:
                ijkToRAS_t1 = vtk.vtkMatrix4x4()
                self._t1_node.GetIJKToRASMatrix(ijkToRAS_t1)
                brain_label_node.SetIJKToRASMatrix(ijkToRAS_t1)

            updateVolumeFromArray(brain_label_node, brain_mask_t1)

            # Inherit any parent transform the T1 carries (e.g. from registration)
            t1_transform = self._t1_node.GetParentTransformNode()
            if t1_transform is not None:
                brain_label_node.SetAndObserveTransformNodeID(t1_transform.GetID())

            # Harden all transforms on brain_label_node so it is in pure world (RAS) space
            slicer.vtkSlicerTransformLogic.hardenTransform(brain_label_node)

            # The resamplescalarvectordwivolume CLI reads only the local IJKToRAS of the
            # reference volume (it does not follow parent transforms).  Harden the CT
            # temporarily so the CLI sees world-space geometry, then fully restore it.
            ct_ijk_to_ras_orig = vtk.vtkMatrix4x4()
            self._ct_node.GetIJKToRASMatrix(ct_ijk_to_ras_orig)
            ct_transform = self._ct_node.GetParentTransformNode()
            ct_transform_id = ct_transform.GetID() if ct_transform is not None else None
            if ct_transform_id is not None:
                slicer.vtkSlicerTransformLogic.hardenTransform(self._ct_node)

            # --- Resample into CT space ---
            brain_label_ct = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "_SEEGFellow_BrainMask_CT"
            )
            params = {
                "inputVolume": brain_label_node.GetID(),
                "referenceVolume": self._ct_node.GetID(),  # hardened → world-space geometry
                "outputVolume": brain_label_ct.GetID(),
                "interpolationMode": "NearestNeighbor",
            }
            slicer.cli.runSync(
                slicer.modules.resamplescalarvectordwivolume, None, params
            )

            # Fully restore CT: reset geometry and re-attach the original transform
            if ct_transform_id is not None:
                self._ct_node.SetIJKToRASMatrix(ct_ijk_to_ras_orig)
                self._ct_node.SetAndObserveTransformNodeID(ct_transform_id)

            brain_mask_in_ct = np.array(arrayFromVolume(brain_label_ct), dtype=np.uint8)
            brain_mask_in_ct = (brain_mask_in_ct > 0).astype(np.uint8)

            print(
                f"[SEEGFellow] Brain mask voxel count in CT space: "
                f"{np.sum(brain_mask_in_ct > 0)}"
            )

            # --- Display as segmentation segment ---
            if self._segmentation_node is None:
                self._segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", "SEEGFellow Segmentation"
                )
                self._segmentation_node.CreateDefaultDisplayNodes()
                self._segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                    self._ct_node
                )

            seg = self._segmentation_node.GetSegmentation()
            existing_id = seg.GetSegmentIdBySegmentName("Brain")
            if existing_id:
                seg.RemoveSegment(existing_id)

            segment_id = seg.AddEmptySegment("Brain", "Brain")
            seg.GetSegment(segment_id).SetColor(0.0, 0.5, 1.0)
            # brain_label_ct carries world-space geometry (its IJKToRAS is the
            # hardened CT geometry), so the stored segment aligns with the CT.
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                brain_mask_in_ct, self._segmentation_node, segment_id, brain_label_ct
            )
            self._head_mask = brain_mask_in_ct
        finally:
            for tmp in (brain_label_node, brain_label_ct):
                if tmp is not None:
                    slicer.mrmlScene.RemoveNode(tmp)

        # Add hemisphere surface segments for 3-D visualization
        from SEEGFellowLib.brain_mask import SynthSegBrainMask

        if isinstance(strategy, SynthSegBrainMask) and self._parcellation is not None:
            self._add_hemisphere_segments()

    def _add_hemisphere_segments(self) -> None:
        """Add Left/Right Hemisphere segments from the SynthSeg parcellation.

        Uses all left- and right-hemisphere DKT labels to build a binary mask
        whose outer 3-D surface shows cortical gyri and sulci. Each hemisphere is
        resampled to CT space using the same pipeline as the Brain segment.
        Segments are set to 50 % 3-D opacity so electrodes remain visible.

        Called automatically at the end of run_intracranial_mask when a
        SynthSegBrainMask strategy was used.

        Example::

            logic.run_intracranial_mask()  # calls _add_hemisphere_segments internally
        """
        import numpy as np
        import vtk
        import slicer
        from slicer.util import arrayFromVolume, updateVolumeFromArray
        from SEEGFellowLib.hemisphere_labels import (
            LEFT_HEMISPHERE_LABELS,
            RIGHT_HEMISPHERE_LABELS,
        )

        hemispheres = [
            ("Left Hemisphere", LEFT_HEMISPHERE_LABELS, (0.6, 0.65, 0.75)),
            ("Right Hemisphere", RIGHT_HEMISPHERE_LABELS, (0.75, 0.65, 0.65)),
        ]

        # Harden CT once before the loop so the CLI sees world-space geometry,
        # then fully restore both IJKToRAS and parent transform afterwards
        # (hardenTransform bakes the parent into IJKToRAS, so restoring just the
        # parent ID without resetting IJKToRAS would double-apply it).
        ct_ijk_to_ras_orig = vtk.vtkMatrix4x4()
        self._ct_node.GetIJKToRASMatrix(ct_ijk_to_ras_orig)
        ct_transform = self._ct_node.GetParentTransformNode()
        ct_transform_id = ct_transform.GetID() if ct_transform is not None else None
        if ct_transform_id is not None:
            slicer.vtkSlicerTransformLogic.hardenTransform(self._ct_node)

        try:
            for name, labels, color in hemispheres:
                mask = np.isin(self._parcellation, sorted(labels)).astype(np.uint8)

                label_node = None
                label_ct = None
                try:
                    # --- Temp labelmap in parcellation (SynthSeg 1 mm) space ---
                    safe_name = name.replace(" ", "")
                    label_node = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLLabelMapVolumeNode", f"_SEEGFellow_{safe_name}_MRI"
                    )
                    mask_mat = vtk.vtkMatrix4x4()
                    for r_idx in range(4):
                        for c_idx in range(4):
                            mask_mat.SetElement(
                                r_idx,
                                c_idx,
                                float(self._parcellation_affine[r_idx, c_idx]),
                            )
                    label_node.SetIJKToRASMatrix(mask_mat)
                    updateVolumeFromArray(label_node, mask)

                    # Inherit and harden T1 parent transform (CT→T1 registration)
                    t1_transform = self._t1_node.GetParentTransformNode()
                    if t1_transform is not None:
                        label_node.SetAndObserveTransformNodeID(t1_transform.GetID())
                    slicer.vtkSlicerTransformLogic.hardenTransform(label_node)

                    # --- Resample to CT space (CT is already hardened above) ---
                    label_ct = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLLabelMapVolumeNode", f"_SEEGFellow_{safe_name}_CT"
                    )
                    params = {
                        "inputVolume": label_node.GetID(),
                        "referenceVolume": self._ct_node.GetID(),
                        "outputVolume": label_ct.GetID(),
                        "interpolationMode": "NearestNeighbor",
                    }
                    slicer.cli.runSync(
                        slicer.modules.resamplescalarvectordwivolume, None, params
                    )

                    hemi_mask = (np.array(arrayFromVolume(label_ct)) > 0).astype(
                        np.uint8
                    )
                    print(
                        f"[SEEGFellow] {name} voxel count in CT space: {hemi_mask.sum()}"
                    )

                    # --- Add segment ---
                    seg = self._segmentation_node.GetSegmentation()
                    existing_id = seg.GetSegmentIdBySegmentName(name)
                    if existing_id:
                        seg.RemoveSegment(existing_id)

                    segment_id = seg.AddEmptySegment(name, name)
                    seg.GetSegment(segment_id).SetColor(*color)
                    slicer.util.updateSegmentBinaryLabelmapFromArray(
                        hemi_mask, self._segmentation_node, segment_id, label_ct
                    )

                    # 50 % opacity so electrodes remain visible through the surface
                    display_node = self._segmentation_node.GetDisplayNode()
                    display_node.SetSegmentOpacity3D(segment_id, 0.5)

                finally:
                    for tmp in (label_node, label_ct):
                        if tmp is not None:
                            slicer.mrmlScene.RemoveNode(tmp)
        finally:
            # Fully restore CT: reset geometry then re-attach the original transform.
            if ct_transform_id is not None:
                self._ct_node.SetIJKToRASMatrix(ct_ijk_to_ras_orig)
                self._ct_node.SetAndObserveTransformNodeID(ct_transform_id)

    def run_metal_threshold(self, threshold: float = 2500) -> None:
        """Threshold CT within intracranial mask and display as a segment.

        Example::

            logic.run_metal_threshold(threshold=2500)
        """
        from SEEGFellowLib.metal_segmenter import threshold_volume
        from slicer.util import arrayFromVolume

        ct_array = arrayFromVolume(self._ct_node)
        metal_mask = threshold_volume(ct_array, threshold)
        if self._head_mask is not None:
            metal_mask = metal_mask & self._head_mask

        seg = self._segmentation_node.GetSegmentation()
        existing_id = seg.GetSegmentIdBySegmentName("Metal")
        if existing_id:
            seg.RemoveSegment(existing_id)

        segment_id = seg.AddEmptySegment("Metal", "Metal")
        seg.GetSegment(segment_id).SetColor(1.0, 1.0, 0.0)

        # updateSegmentBinaryLabelmapFromArray uses only the local IJKToRAS of the
        # reference node (parent transforms are not followed).  Harden the CT
        # temporarily so the segment is stored in world space, matching the brain mask.
        import vtk

        ct_ijk_to_ras_orig = vtk.vtkMatrix4x4()
        self._ct_node.GetIJKToRASMatrix(ct_ijk_to_ras_orig)
        ct_transform = self._ct_node.GetParentTransformNode()
        ct_transform_id = ct_transform.GetID() if ct_transform is not None else None
        if ct_transform_id is not None:
            slicer.vtkSlicerTransformLogic.hardenTransform(self._ct_node)

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            metal_mask, self._segmentation_node, segment_id, self._ct_node
        )

        if ct_transform_id is not None:
            self._ct_node.SetIJKToRASMatrix(ct_ijk_to_ras_orig)
            self._ct_node.SetAndObserveTransformNodeID(ct_transform_id)

        self._metal_mask = metal_mask

    def run_electrode_detection(
        self,
        sigma: float = 1.2,
        expected_spacing: float = 3.5,
        min_contacts: int = 3,
        max_component_voxels: int = 500,
        spacing_cutoff_factor: float = 0.65,
        distance_tolerance: float = 1.5,
        max_iterations: int = 1000,
    ) -> None:
        """Run LoG contact detection, group into electrodes, and place fiducials.

        Step 4b (metal threshold) must be completed before calling this method.

        Example::

            logic.run_electrode_detection(sigma=1.2)
        """
        import numpy as np
        from SEEGFellowLib.electrode_detector import ElectrodeDetector

        if self._metal_mask is None:
            raise RuntimeError(
                "Metal mask not computed. Run step 4b (metal threshold) first."
            )

        # Remove fiducials from any previous run.
        if self._raw_contacts_node_id is not None:
            node = slicer.mrmlScene.GetNodeByID(self._raw_contacts_node_id)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
            self._raw_contacts_node_id = None
        self.electrodes = []

        # Compute brain centroid from head mask (used for deepest-first orientation).
        brain_centroid = None
        if self._head_mask is not None:
            brain_voxels = np.argwhere(self._head_mask > 0)
            if len(brain_voxels) > 0:
                # brain_voxels rows are (K, J, I); reverse to get IJK
                centroid_ijk = brain_voxels.mean(axis=0)[::-1]  # KJI → IJK
                ijk_to_ras = ElectrodeDetector._get_ijk_to_ras_matrix(self._ct_node)
                centroid_h = np.append(centroid_ijk, 1.0)
                brain_centroid = (ijk_to_ras @ centroid_h)[:3]

        detector = ElectrodeDetector(
            min_contacts=min_contacts,
            expected_spacing=expected_spacing,
            distance_tolerance=distance_tolerance,
            max_iterations=max_iterations,
            spacing_cutoff_factor=spacing_cutoff_factor,
        )

        self.electrodes = detector.detect_all(
            self._ct_node,
            self._metal_mask,
            sigma=sigma,
            max_component_voxels=max_component_voxels,
            brain_centroid=brain_centroid,
        )
        print(f"[SEEGFellow] Detected {len(self.electrodes)} electrodes")

        self._create_fiducials_for_electrodes()

    def run_contact_labeling(self) -> None:
        """Label each contact with its anatomical region from the SynthSeg parcellation.

        Requires parcellation (from run_intracranial_mask with SynthSeg) and
        detected electrodes.

        Example::

            logic.run_contact_labeling()
            for e in logic.electrodes:
                for c in e.contacts:
                    print(c.label, c.region)
        """
        import numpy as np
        from SEEGFellowLib.contact_labeler import label_contacts

        if self._parcellation is None:
            raise RuntimeError(
                "No parcellation available. Run brain segmentation (SynthSeg) first."
            )
        if not self.electrodes:
            raise RuntimeError("No electrodes detected. Run electrode detection first.")

        for electrode in self.electrodes:
            contacts_ras = np.array([c.position_ras for c in electrode.contacts])
            regions = label_contacts(
                contacts_ras, self._parcellation, self._parcellation_affine
            )
            for contact, region in zip(electrode.contacts, regions):
                contact.region = region

    # 20 visually distinct colors for electrode fiducials (tab20-inspired)
    ELECTRODE_COLORS = [
        (0.122, 0.467, 0.706),  # blue
        (1.000, 0.498, 0.055),  # orange
        (0.173, 0.627, 0.173),  # green
        (0.839, 0.153, 0.157),  # red
        (0.580, 0.404, 0.741),  # purple
        (0.549, 0.337, 0.294),  # brown
        (0.890, 0.467, 0.761),  # pink
        (0.498, 0.498, 0.498),  # gray
        (0.737, 0.741, 0.133),  # olive
        (0.090, 0.745, 0.812),  # cyan
        (0.682, 0.780, 0.910),  # light blue
        (1.000, 0.733, 0.471),  # light orange
        (0.596, 0.875, 0.541),  # light green
        (1.000, 0.596, 0.588),  # light red
        (0.773, 0.690, 0.835),  # light purple
        (0.769, 0.612, 0.580),  # light brown
        (0.969, 0.714, 0.824),  # light pink
        (0.780, 0.780, 0.780),  # light gray
        (0.859, 0.859, 0.553),  # light olive
        (0.620, 0.855, 0.898),  # light cyan
    ]

    def _create_fiducials_for_electrodes(self) -> None:
        """Create a markups fiducial node for each electrode's contacts."""
        for idx, electrode in enumerate(self.electrodes):
            # Default label uses dash prefix: "-1", "-2", ...
            if not electrode.name:
                electrode.assign_labels("-")
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode",
                electrode.name if electrode.name else "-",
            )
            for contact in electrode.contacts:
                r, a, s = contact.position_ras
                node.AddControlPoint(r, a, s, contact.label)
            electrode.markups_node_id = node.GetID()

            # Display properties
            display = node.GetDisplayNode()
            color = self.ELECTRODE_COLORS[idx % len(self.ELECTRODE_COLORS)]
            display.SetSelectedColor(*color)
            display.SetColor(*color)
            display.SetOpacity(0.7)
            display.SetGlyphScale(2.0)
            display.SetTextScale(2.25)
            display.SetUseGlyphScale(True)

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
    # Results & Export
    # -------------------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """Export all contact positions and regions to a CSV file.

        Example::

            logic.export_csv("/output/contacts.csv")
        """
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Electrode", "Contact", "R", "A", "S", "Region"])
            for electrode in self.electrodes:
                for contact in electrode.contacts:
                    r, a, s = contact.position_ras
                    writer.writerow(
                        [electrode.name, contact.label, r, a, s, contact.region]
                    )

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
