#ifndef RAYOPTIX_H
#define RAYOPTIX_H

#ifdef  __cplusplus
extern "C" {
#endif

void trace_origin_direction(
  const float, const float, const float,
    const float*, const int,
    const int*, const int,  
    const float*, const float*, 
    const int, float*);
#ifdef  __cplusplus
}
#endif

#endif  /* RAYOPTIX_H */  

