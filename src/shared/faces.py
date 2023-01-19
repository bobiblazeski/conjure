# pyright: reportMissingImports=false
import torch

from trimesh.util import triangle_strips_to_faces

def create_strips(n, m):
    res = []
    for i in range(n-1):
        strip = []
        for j in range(m):            
            strip.append(j+(i+1)*m)
            strip.append(j+i*m)
            #strip.append(j+(i+1)*m)
        res.append(strip)
    return res

def make_faces(n, m):
    strips = create_strips(n, m)    
    return triangle_strips_to_faces(strips)


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

edges = torch.tensor([
    [ 0, 22,  2],
    [ 2, 22, 23],#
    [ 3,  7,  2],
    [ 7,  2,  6],#   
    [ 1, 18, 19],
    [ 1, 19,  3],#    
    [ 1, 14, 15],
    [ 1, 14,  0],#    
    [13, 15, 16],
    [15, 16, 18],#    
    [12, 14, 20],
    [14, 22, 20],#    
    [23, 21,  6],
    [ 6,  4, 21],#    
    [ 7, 19, 17],
    [ 5,  7, 17],#    
    [11, 16, 17],
    [11,  9, 16],#
    [ 8,  9, 13],
    [13,  8, 12],#
    [10,  4,  5],
    [11,  5, 10],#    
    [10,  8, 21],
    [20, 21,  8],
])

def get_edge_vertices(tri, pair, n):
    square = torch.arange(n**2).reshape(n, n)
    if pair == [0, 1]:
        r = square[0, :]    
    elif pair == [0, 2]:
        r = square[:, 0]
    elif pair == [1, 3]:
        r = square[:, -1]
    elif pair == [2, 3]:
        r = square[-1, :]
    else:
        raise Exception(f'Unknown pair {pair}')
    return ((n ** 2) * tri  + r).tolist()

def single_edge_faces(l1, l2):
    t1 = [[a, b, c] for a, b, c 
          in zip(l1, l1[1:], l2)]
    t2 = [[a, b, c] for a, b, c 
          in zip(l2, l2[1:], l1[1:])]
    return t1 + t2

def make_edges(n):
    res  = []
    for x, y in pairwise(edges.tolist()):
        m = list(set(x + y))
        m.sort()
        a1, a2, b1, b2 = m              
        pair_a = [a1 % 4, a2 % 4]
        pair_b = [b1 % 4, b2 % 4]         
        tri_a, tri_b = a1 // 4, b1 // 4                
        l1 = get_edge_vertices(tri_a, pair_a, n)
        l2 = get_edge_vertices(tri_b, pair_b, n)        
        res = res + single_edge_faces(l1, l2)     
    return torch.tensor(res)

def make_corners(n):
    s0 = 0
    s1 = n-1
    s2 = n**2-n
    s3 = n**2-1
    tris = torch.tensor([
        [0, 5, 3],
        [0, 4, 1],
        [4, 2, 3],
        [4, 2, 1],
        [0, 4, 3],
        [0, 5, 1],
        [5, 2, 3],
        [5, 2, 1]])  
    rmn = torch.tensor([
        [s0, s2, s2],
        [s3, s3, s3],
        [s0, s1, s1],
        [s1, s3, s1],
        [s1, s2, s3],
        [s2, s3, s2],
        [s0, s0, s0],
        [s1, s2, s0]])
    return tris * n**2+ rmn

def make_sides(n):
    offset, faces = n ** 2, make_faces(n, n)
    sides = torch.cat([
        i * offset + torch.tensor(faces)
        for i in range(6)])
    return sides

def make_cube_faces(n):
    sides = make_sides(n)
    corners = make_corners(n)
    edges = make_edges(n)
    return torch.cat((sides, corners, edges)).int()



def make_cylinder_faces(m, n):
    def closing_faces(rows, cols):    
      strips = []
      for row in range(rows-1):        
          strips.append([
              (row+1)*cols+cols-1,
              row*cols+cols-1,
              row*cols+cols,
              row*cols
          ])      
      return triangle_strips_to_faces(strips)

    main = torch.tensor(make_faces(m, n))
    closing = torch.tensor(closing_faces(m, n))    
    return torch.cat((main, closing)).int()