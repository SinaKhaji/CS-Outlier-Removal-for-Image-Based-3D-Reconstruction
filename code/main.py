import os
from utils import load_views, load_bunny, render_views, save_all_points, denoise, poisson, view

DATASETS = ['bunny', 'teddy']
POISSON_D = 8

BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUT         = os.path.join(RESULTS_DIR, 'out')

if __name__ == '__main__':
    samples = {}
    for ds in DATASETS:
        OUT_DS            = os.path.join(OUT, ds)
        FUSED_DS          = os.path.join(OUT_DS, 'fused.ply')
        FUSED_FILTERED_DS = os.path.join(OUT_DS, 'fused_filtered.ply')
        MESH_DS           = os.path.join(OUT_DS, 'mesh.ply')
        MESH_FILTERED_DS  = os.path.join(OUT_DS, 'mesh_filtered.ply')
        os.makedirs(OUT_DS, exist_ok=True)

        if ds == 'teddy':
            views = load_views(os.path.join(DATA_DIR, 'teddy'))
        else:
            mesh = load_bunny(os.path.join(DATA_DIR, 'bunny'))
            views = render_views(mesh)
        print(ds, len(views), "views loaded")

        save_all_points(views, FUSED_DS)

        denoise(views, FUSED_FILTERED_DS)

        mesh = poisson(FUSED_DS, MESH_DS, POISSON_D)
        mesh_filtered = poisson(FUSED_FILTERED_DS, MESH_FILTERED_DS, POISSON_D)
        samples[ds+"_unfiltered"] = mesh
        samples[ds+"_filtered"] = mesh_filtered
    view(samples)