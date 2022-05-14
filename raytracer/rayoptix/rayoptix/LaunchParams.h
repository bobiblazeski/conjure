#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
  using namespace gdt;
  
  struct TriangleMeshSBTData {
    vec3f *vertex;
    vec3i *index;
  };
  
  struct RotationMatrix{
      float r0c0;
      float r0c1;
      float r0c2;
      float r1c0;
      float r1c1;
      float r1c2;
      float r2c0;
      float r2c1;
      float r2c2;
  };

  struct LaunchParams
  {
    struct {
      float *distanceBuffer;
      float *angleBuffer;
      float *rayBuffer;
      float *directionBuffer;
      float *pointBuffer;
      float *triangleIndicesBuffer;
      vec2i     size;
    } frame;                

    float rayOriginX;
    float defaultDistance;
    float defaultAngle;
    float tmax;
    RotationMatrix rotationMatrix;

    OptixTraversableHandle traversable;
  };

} // ::osc
