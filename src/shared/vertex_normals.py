# pyright: reportMissingImports=false
import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from src.shared.faces import make_cube_faces

def vertex_tris(faces):
    res = [[] for _ in range(faces.max()+1)]
    for fid, face in enumerate(faces):        
        for vid in face:
            res[vid].append(fid)        
    return res

def vertex_tri_maps(faces):
    vts = vertex_tris(faces)
    r, c = len(vts), max([len(x) for  x in vts])
    vert_tri_indices = torch.zeros(r, c, dtype=torch.long)
    vert_tri_weights = torch.zeros(r, c)    
    for r, tris in enumerate(vts):        
        weight = 1. #/ len(tris)
        for c, tri_id in enumerate(tris):
            vert_tri_indices[r, c] = tri_id
            vert_tri_weights[r, c] = weight
    return vert_tri_indices, vert_tri_weights.unsqueeze(dim=-1)

class VertexNormals(torch.nn.Module):
    
    def __init__(self, opt, load=True, size=None):
        super().__init__()
        self.size = size or opt.adversarial_data_patch_size
        self.path = os.path.join(opt.data_dir, 
            'trimap/trimap_{}.pth'.format(self.size))
        if load and os.path.exists(self.path):
            trimap = torch.load(self.path)
        else:
            print('Creating trimap this might take long time.')
            trimap = self.make_trimap(self.size)
            torch.save(trimap, self.path)
        self.assign_trimap(trimap)
    
    def assign_trimap(self,  trimap):        
        self.register_buffer('vert_tri_indices', trimap['vert_tri_indices'])
        self.register_buffer('vert_tri_weights', trimap['vert_tri_weights'])        

    def get_face_normals(self, vrt, faces, normalized=True):       
        v1 = vrt.index_select(0, faces[:, 1]) - vrt.index_select(0, faces[:, 0])
        v2 = vrt.index_select(0, faces[:, 2]) - vrt.index_select(0, faces[:, 0])
        face_normals = v1.cross(v2) # [F, 3]
        if normalized:
            face_normals = F.normalize(face_normals, p=2, dim=-1) # [F, 3]
        return face_normals.t()

    def vertex_normals_fast(self, vrt, faces, normalized=True):
        face_normals = self.get_face_normals(vrt, faces, normalized=False)    
        r, c = self.vert_tri_indices.shape    
        vert_tri_indices, vert_tri_weights = vertex_tri_maps(faces)        
        fn_group = face_normals.index_select(1, 
            vert_tri_indices.flatten()).reshape(r, c, 3)
        weighted_fn_group = fn_group * vert_tri_weights    
        vertex_normals = weighted_fn_group.sum(dim=-2)
        if normalized:
            return F.normalize(vertex_normals, p=2, dim=-1)
        return vertex_normals

        
    def __repr__(self):
        return f'VertexNormals: size: {self.size} path: {self.path}'
    
    def make_trimap(self, size):
        faces = make_cube_faces(size)
        vert_tri_indices, vert_tri_weights = vertex_tri_maps(faces)        
        return OrderedDict([
          ('vert_tri_indices', vert_tri_indices),
          ('vert_tri_weights', vert_tri_weights),                 
        ])
    