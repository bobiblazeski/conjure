project(course)

cmake_minimum_required(VERSION 2.8)

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

#set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
#include(${gdt_dir}/cmake/configure_glfw.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
find_package(glfw3 REQUIRED)

include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)


# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

set(optix_LIBRARY "")

##########################################################################
#  Build the library 
##########################################################################

#add_subdirectory(raytriangle EXCLUDE_FROM_ALL)

##################
#  Build your example Apps
############
#add_subdirectory(executable)

add_subdirectory(rayoptix EXCLUDE_FROM_ALL)
#add_subdirectory(example05)

add_subdirectory(tryoptix)


