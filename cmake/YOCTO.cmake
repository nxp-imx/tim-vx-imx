set(PKG_NAME "OVXDRV")
message("Using YOCTO Project configuration. Toolchain and SDK expected to be set in TOOLCHAIN")

# The include directories are available in SDK
set(OVXDRV_INCLUDE_DIRS)

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    libCLC.so
    libGAL.so
    libOpenVX.so
    libOpenVXU.so
    libVSC.so
    libArchModelSw.so
    libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})