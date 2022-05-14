import ctypes
import torch
import numpy as np


def to_np(t):
    return t.cpu().detach().numpy()

def np_flat(arr, dtype=np.float32):
    return arr.flatten().astype(dtype)


class Tracer:
    """ Trace generated mesh over a template mesh.
    The rays are coming from generated mesh vertices toward
    the center of a sphere [0.0, 0.0, 0.].
    """    
    def __init__(self, lib_dir):
        """
        Args:
            lib_dir (str): path of the compiled c library.
        """
        self.lib_dir = lib_dir        
        self.__trace_origin_direction = self.__create_trace_origin_direction(lib_dir)
    
    def __repr__(self):
        return "Tracer lib_dir:{}".format(self.lib_dir)

    def __create_trace_origin_direction(self, lib_dir):
        """
        Create FFI binding to the trace_normals function.
        
        Args:
            lib_dir (str): path of the compiled c library.
        """
        librayoptix = ctypes.CDLL(lib_dir+'librayoptix.so')

        trace_normals  = librayoptix.__getattr__('trace_origin_direction')
        trace_normals.restype = None
        trace_normals.argtypes = [        
            ctypes.c_float, # defaultDistance
            ctypes.c_float, # defaultAngle
            ctypes.c_float, # tmax
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  
            ctypes.c_int, # no_vertices_x3 
            np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ctypes.c_int,# no_faces_x3        
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_int, # pixel_no          
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),            
        ]
        return trace_normals        
    
    def __call__(self, ray_origins_, ray_directions_,
          vertices_, faces_, defaultDistance, defaultAngle, tmax):
        """ Raytrace using RTX cores.
        
        Arguments:
            ray_origins_ (Tensor [n, 3]): ray origins 
            ray_directions_ (Tensor [n, 3]): ray directions
            vertices_ (Tensor [n, 3]): mesh vertices
            faces_ (Tensor [n, 3]): mesh faces            
            defaultDistance(float): distance returned on ray miss
            defaultAngle(float): angle returned on ray miss
            tmax (float) - distance for ray propagation
            
        """
        ray_origins, ray_directions, vertices, faces = [
            x.reshape(-1, faces_.size(-1)) for x in [ray_origins_, 
                ray_directions_, vertices_, faces_]]        
        # Ctypes
        defaultDistance = ctypes.c_float(defaultDistance)
        defaultAngle = ctypes.c_float(defaultAngle)
        tmax = ctypes.c_float(tmax)
        
        # Mesh vertices & faces
        verticesFlat = np_flat(to_np(vertices))  
        noVerticesX3 = ctypes.c_int(verticesFlat.shape[0])
        facesFlat = np_flat(to_np(faces), dtype=np.int32)
        noFacesX3 = ctypes.c_int(facesFlat.shape[0])
        
        # Rays origins & directions
        pixel_no = ray_origins.size(0)
        rayOriginsFlat = np_flat(to_np(ray_origins))
        rayDirectionsFlat = np_flat(to_np(ray_directions))
        assert len(rayOriginsFlat) == len(rayDirectionsFlat)        
                
        pixel_no_c = ctypes.c_int(pixel_no)
        
        result_points = np_flat(np.empty((5*pixel_no,)))        
        
        self.__trace_origin_direction(defaultDistance, defaultAngle, tmax,
           verticesFlat, noVerticesX3, facesFlat, noFacesX3, 
           rayOriginsFlat, rayDirectionsFlat, pixel_no_c,
           result_points)
        
        return torch.from_numpy(result_points).reshape(ray_origins_.size(0), -1, 5)
