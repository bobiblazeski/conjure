#include <optix_device.h>

#include "LaunchParams.h"

using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
    
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // Get triangle vertices
    const unsigned int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];    

    const float2 bary = optixGetTriangleBarycentrics();
    const float u = bary.x;
    const float v = bary.y;
    const float w = 1. - u - v;
    // Convert barycentric to cartesian coordinates for x axis
    const float p_x = u*A.x + v*B.x + w*C.x;
    const float p_y = u*A.y + v*B.y + w*C.y;
    const float p_z = u*A.z + v*B.z + w*C.z;
    // Assign cartesian x to payload        
    optixSetPayload_0(__float_as_uint(p_x));
    optixSetPayload_1(__float_as_uint(p_y));
    optixSetPayload_2(__float_as_uint(p_z));

    // compute normal:
    const vec3f Ng     = normalize(cross(B-A,C-A));
    const vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,Ng));
    
    optixSetPayload_3(__float_as_uint(cosDN));

    //const uint32_t p4 = 12345;
    //const unsigned int p4 = 10203040;
    // This is weird if primID is uint but must be converted 
    // to float or returns garbage ????!!!!
    optixSetPayload_4(__float_as_uint(primID));
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */}
 
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  extern "C" __global__ void __miss__radiance()
  { /* Leave default on miss */}

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __raygen__renderNormals()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    
    // our per-ray data for this example. 
    uint32_t u0 = __float_as_uint(optixLaunchParams.defaultDistance);
    uint32_t u1 = __float_as_uint(optixLaunchParams.defaultDistance);
    uint32_t u2 = __float_as_uint(optixLaunchParams.defaultDistance);
    uint32_t u3 = __float_as_uint(optixLaunchParams.defaultAngle);
    uint32_t u4 = __float_as_uint(-111.);
    
    // generate ray direction
    const uint32_t rayIndex = (ix+iy*optixLaunchParams.frame.size.x)*3;
    vec3f rayOrigin = vec3f(optixLaunchParams.frame.rayBuffer[rayIndex+0],
                            optixLaunchParams.frame.rayBuffer[rayIndex+1],
                            optixLaunchParams.frame.rayBuffer[rayIndex+2]);

    vec3f rayDirection = normalize(
      vec3f(optixLaunchParams.frame.directionBuffer[rayIndex+0],
            optixLaunchParams.frame.directionBuffer[rayIndex+1],
            optixLaunchParams.frame.directionBuffer[rayIndex+2]));

    optixTrace(optixLaunchParams.traversable,
               rayOrigin, //camera.position,
               rayDirection,
               0.f,    // tmin
               optixLaunchParams.tmax,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1, u2, u3, u4);

    // and write to frame buffer ...   
    const uint32_t resIndex = (ix+iy*optixLaunchParams.frame.size.x)*5; 
    optixLaunchParams.frame.pointBuffer[resIndex+0] =  __uint_as_float(u0);
    optixLaunchParams.frame.pointBuffer[resIndex+1] =  __uint_as_float(u1);
    optixLaunchParams.frame.pointBuffer[resIndex+2] =  __uint_as_float(u2);
    optixLaunchParams.frame.pointBuffer[resIndex+3] =  __uint_as_float(u3);
    optixLaunchParams.frame.pointBuffer[resIndex+4] =  __uint_as_float(u4);
  }
  
}

inline __device__ vec3f rotate_vec3f(vec3f v, RotationMatrix m) { 
  return vec3f(v.x*m.r0c0+v.y*m.r1c0+v.z*m.r2c0,
               v.x*m.r0c1+v.y*m.r1c1+v.z*m.r2c1,
               v.x*m.r0c2+v.y*m.r1c2+v.z*m.r2c2);
}

// ::osc
