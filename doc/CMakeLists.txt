# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Build Doxygen documentation
SET(DOXYGEN_MIN_VERSION "1.8.5")
find_package(Doxygen ${DOXYGEN_MIN_VERSION} REQUIRED)

set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/indexl.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

# Create Doxyfile
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating Doxygen documentation")

add_custom_target(docs ALL DEPENDS ${DOXYGEN_INDEX_FILE}) 

install(DIRECTORY
    ${CMAKE_BINARY_DIR}/doc/doxygen
    DESTINATION doc)
