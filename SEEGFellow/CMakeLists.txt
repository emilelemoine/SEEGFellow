cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)
project(SEEGFellow)

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(SEEGFellow)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
