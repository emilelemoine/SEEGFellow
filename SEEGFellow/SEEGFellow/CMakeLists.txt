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
