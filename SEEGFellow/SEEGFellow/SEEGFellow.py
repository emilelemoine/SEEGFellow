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
