# conjure
Create shape from image


The GOAL

Fit single mesh from rendering image

Use coarse to fit:

1 Fit Ellipsoid
  -  Start from scratch

2 Fit Geoid
  - Use upscaled ellipsoid result as an input

3 Fit Coarse Manifold
  - Use upscaled geoid as an input
  
4 Fit Finer Manifold
  - Use upscaled manifold as an input