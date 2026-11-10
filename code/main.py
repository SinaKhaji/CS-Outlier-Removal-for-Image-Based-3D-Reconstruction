import os
from utils import load_views, poisson, view, denoise

POISSON_D = 8

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUT         = os.path.join(DATA_DIR, 'out')
MVS         = os.path.join(OUT, 'mvs')
IMG_DIR     = os.path.join(MVS, 'images')
FUSED         = os.path.join(OUT, 'fused.ply')
FUSED_FILTERED      = os.path.join(OUT, 'fused_filtered.ply')
MESH          = os.path.join(OUT, 'mesh.ply')
MESH_FILTERED = os.path.join(OUT, 'mesh_filtered.ply')

if __name__ == '__main__':
    views = load_views(MVS)
    n_views = len(views)
    print(n_views, "views loaded")

    denoise(views, FUSED_FILTERED)

    mesh = poisson(FUSED, MESH, POISSON_D)
    mesh_filtered = poisson(FUSED_FILTERED, MESH_FILTERED, POISSON_D)

    view(mesh, mesh_filtered)