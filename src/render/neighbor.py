import torch
from sklearn.neighbors import NearestNeighbors

class Neighbor:
    """Find nearest neighbor by triangle index."""
    
    def __init__(self, faces, vertices, n_neighbors):        
        index= faces.flatten()
        barycenters = vertices.index_select(0, 
           index).reshape(-1, 3, 3).sum(dim=1)/3
        neigh = NearestNeighbors(n_neighbors=1).fit(vertices)
                
        self.n_neighbors = n_neighbors
        self.face_no = faces.size(-2)
        self.vertex_no = vertices.size(-2)
        self.neighbor_indices = torch.from_numpy(
            neigh.kneighbors(barycenters, 
            n_neighbors=n_neighbors, return_distance=False)
        ).long()                
        
    def __call__(self, hit_face_indices, vertices):
        # Replace misses (-1s) with random triangles
        bs, px_no = hit_face_indices.shape
        vertex_indices_all = torch.zeros((bs, px_no * self.n_neighbors)).long()
        for i, hfi in enumerate(hit_face_indices):
            face_idx_offset = i * self.face_no            
            reverted_hfi = hfi-face_idx_offset
            augmented = torch.where(reverted_hfi < 0, 
              torch.randint_like(hfi, low=0, high=self.face_no),
              reverted_hfi)
            indices = self.neighbor_indices.index_select(0, augmented)
            vertex_indices = self.neighbor_indices.index_select(0, augmented)            
            vertex_indices += i * self.vertex_no # Offset vertex indices
            vertex_indices_all[i] = vertex_indices.flatten()
        vertex_indices_all = vertex_indices_all.flatten().to(vertices.device)
        npts = vertices.reshape(-1, 3).index_select(0, vertex_indices_all)       
        return npts.reshape(bs, px_no, self.n_neighbors, 3)
    
    def __repr__(self):
        return "Neighbor n_neighbors:{}, face_no:{}, vertex_no:{}".format(            
            self.n_neighbors, self.face_no, self.vertex_no)