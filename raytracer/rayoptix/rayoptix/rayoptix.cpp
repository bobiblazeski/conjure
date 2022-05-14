//#include <string.h> 
#include<algorithm>
#include "SampleRenderer.h"
#include "rayoptix.h"


extern "C" void trace_origin_direction( 
    const float defaultDistance, const float defaultAngle,
    const float tmax,    
    const float* vertices, const int no_vertices_x3,
    const int* faces, const int no_faces_x3,  
    const float* origins, const float* directions,
    const int no_rays, float* res_points) {
  const int no_rays_x3 = no_rays * 3;
  osc::TriangleMesh model;
  // Store vertices
  for (int i = 0; i < no_vertices_x3; i+=3) {
    model.vertex.push_back(osc::vec3f(*(vertices+i), 
      *(vertices+i+1), *(vertices+i+2)));
  }  
  // Store indices
  for (int i = 0; i < no_faces_x3; i+=3) {
    model.index.push_back(osc::vec3i(*(faces+i), 
      *(faces+i+1), *(faces+i+2)));
  }

  osc::vec2i fbSize = osc::vec2i(no_rays, 1);
  osc::SampleRenderer sample(model, defaultDistance, defaultAngle, tmax, osc::TraceFunction::normals);

  std::vector<osc::vec3f> origins_vector;
  std::vector<osc::vec3f> directions_vector;
  for (int i = 0; i < no_rays_x3; i+=3) {
    origins_vector.push_back(osc::vec3f(*(origins+i), 
      *(origins+i+1), *(origins+i+2)));

    directions_vector.push_back(osc::vec3f(*(directions+i), 
      *(directions+i+1), *(directions+i+2)));
  }
        
  sample.resize(fbSize, origins_vector, directions_vector);
  std::vector<float> points;
  points.resize(fbSize.x*fbSize.y*5);  

  sample.render();  
  sample.downloadPoints(points.data());
    
  std::copy_n(points.data(), no_rays * 5, res_points);  
}