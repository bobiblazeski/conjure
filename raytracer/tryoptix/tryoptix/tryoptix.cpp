
#include "rayoptix/rayoptix.h"
#include <vector>
#include <iostream>
  

int main() {
  const float rayOriginX = 2.f;
  const float defaultDistance = 1.f;
  const float defaultAngle = 0.2f;

  const int w = 4;
  const int h = 5;
  const int no_vertices_x3 = w * h *3;
  const float vertices[] = {
    0, 30, 60,  1, 31, 61,  2, 32, 62,  3, 33, 63,  4, 34, 64,  
    5, 35, 65,  6, 36, 66,  7, 37, 67,  8, 38, 68,  9, 39, 69, 
    10, 40, 70, 11, 41, 71, 12, 42, 72, 13, 43, 73, 14, 44, 74, 
    15, 45, 75, 16, 46, 76, 17, 47, 77, 18, 48, 78, 19, 49, 79
  };
  const int resolution = 1024;
  const int length = resolution * resolution;
  

  std::vector<int> faces;
  for (int iw = 0; iw < w-1; ++iw) {
    for (int ih = 0; ih < h-1; ++ih) {
      faces.push_back(iw*w+ih);
      faces.push_back(iw*w+ih+1);
      faces.push_back((iw+1)*w+ih);
      
      faces.push_back(iw*w+ih+1);
      faces.push_back((iw+1)*w+ih);
      faces.push_back((iw+1)*w+ih+1);
    }           
  }             
  const int no_faces_x3 = faces.size();
  const float rotationMatrix[] = {
    1., 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.
  }; 
  float* res_distances = new float[length];
  float* res_angles = new float[length];

  for (int i = 0; i < 2000; ++i) {
    std::cout<< i << " th Call " << std::endl;
    // trace_x(rayOriginX, defaultDistance, defaultAngle, vertices, no_vertices_x3, 
    //   faces.data(), no_faces_x3, resolution, rotationMatrix, 
    //   res_distances, res_angles);
  }
  return 0;
}
