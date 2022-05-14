project(rayoptix_library)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code rayoptix/devicePrograms.cu)

add_library(rayoptix SHARED
  ${embedded_ptx_code}
  rayoptix/optix7.h
  rayoptix/CUDABuffer.h
  rayoptix/SampleRenderer.h
  rayoptix/SampleRenderer.cpp
  rayoptix/rayoptix.h
  rayoptix/rayoptix.cpp
  )

target_link_libraries(rayoptix
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}  
  )
