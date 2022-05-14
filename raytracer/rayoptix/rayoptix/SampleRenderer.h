#pragma once

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct Camera {
    /*! camera position - *from* where we are looking */
    vec3f from;
    /*! which point we are looking *at* */
    vec3f at;
    /*! general up-vector */
    vec3f up;
  };
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {    
    std::vector<vec3f> vertex;
    std::vector<vec3i> index;    
  };
 
  enum TraceFunction { x=0, cylinder=1, normals=2 };
  /*! a sample OptiX-7 renderer that demonstrates how to set up
      context, module, programs, pipeline, SBT, etc, and perform a
      valid launch that renders some pixel (using a simple test
      pattern, in this case */
  class SampleRenderer
  {
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
  public:
    /*! constructor - performs all setup, including initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    SampleRenderer(const TriangleMesh&, const float, const float, const float, const float, const float*);

    SampleRenderer(const TriangleMesh&, const float, const float, const float, const TraceFunction);

    ~SampleRenderer(); // Destructor
    /*! render one frame */
    void render();

    /*! resize frame buffer to given resolution */    
    void resize(const vec2i &newSize, std::vector<vec3f> rays, std::vector<vec3f> directions);

    /*! download the rendered distance buffer */
    void downloadPixels(float h_pixels[], float a_pixels[]);

    /*! download the rendered points buffer */
    void downloadPoints(float h_points[]);

    /*! download the rendered points buffer */
    void downloadTriangleIndices(float h_triangle_indices[]);

    
  protected:
    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! helper function that initializes optix and checks for errors */
    void initOptix();
  
    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModule();
    
    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms(const char*);
    
    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();
    
    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildAccel(const TriangleMesh &model);

  protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipelineLinkOptions    pipelineLinkOptions;
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions;
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /*! @} */

    CUDABuffer distanceBuffer;
    CUDABuffer angleBuffer;
    CUDABuffer rayBuffer;
    CUDABuffer directionBuffer;
    CUDABuffer pointBuffer;    
    
    /*! the model we are going to trace rays against */
    const TriangleMesh model;
    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;
  };

} // ::osc
