cmake_minimum_required (VERSION 3.3)
project(tim-vx LANGUAGES C CXX VERSION 1.1.30.2)

option(TIMVX_REPLACE_COMPILER_FLAGS "Replace standard CMAKE_CXX_FLAGS and CMAKE_C_FLAGS flags." ON)

if(${TIMVX_REPLACE_COMPILER_FLAGS})
    set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror")
endif(${TIMVX_REPLACE_COMPILER_FLAGS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
message(${OVXLIB_API_ATTR})
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})

if("${CONFIG}" STREQUAL "A311D")
    include(cmake/A311D.cmake)
elseif("${CONFIG}" STREQUAL "S905D3")
    include(cmake/S905D3.cmake)
elseif("${CONFIG}" STREQUAL "YOCTO")
    include(cmake/YOCTO.cmake)
else()
    include(cmake/X86_64.cmake)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_subdirectory("src/tim/vx")
add_subdirectory("samples/lenet")
