"""CT-to-T1 rigid registration using BRAINSFit.

Example (in Slicer Python console)::

    reg = CTtoT1Registration()
    transform = reg.run(ct_node, t1_node)
"""

from __future__ import annotations


class CTtoT1Registration:
    """Wraps BRAINSFit for rigid CT-to-T1 registration."""

    def create_rough_transform(self, ct_volume_node):
        """Create a linear transform node and apply it to the CT volume.

        The user will manually adjust this transform to roughly align CT to T1
        before running automated registration.

        Returns:
            vtkMRMLLinearTransformNode applied to the CT volume.
        """
        import slicer

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
        import slicer

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

        cli_node = slicer.cli.runSync(slicer.modules.brainsfit, None, params)

        if cli_node.GetStatus() & cli_node.ErrorsMask:
            error_text = cli_node.GetErrorText()
            raise RuntimeError(f"BRAINSFit registration failed: {error_text}")

        # Apply result transform to CT
        ct_node.SetAndObserveTransformNodeID(output_transform.GetID())

        return output_transform

    def harden_transform(self, volume_node):
        """Harden the transform on a volume node (bake it into the volume)."""
        import slicer

        slicer.vtkSlicerTransformLogic().hardenTransform(volume_node)
