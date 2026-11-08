import os
import shutil
import pycolmap

# ---------- paths (relative to the project root) ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENE = os.path.join(ROOT, "data", "Ammonite")
IMAGES = os.path.join(SCENE, "images")
DB = os.path.join(SCENE, "database.db")
SPARSE = os.path.join(SCENE, "sparse")
DENSE = os.path.join(SCENE, "dense")
MAX_IMAGE_SIZE = 800

print("pycolmap version:", pycolmap.__version__)

# ---------- check the input ----------
if not os.path.isdir(IMAGES):
    raise SystemExit(f"Put your photos in: {IMAGES}")

exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
photos = [f for f in os.listdir(IMAGES) if f.endswith(exts)]
print(f"Found {len(photos)} photos")
if len(photos) < 10:
    raise SystemExit("Too few photos (need at least ~15). HEIC files must be converted to JPG.")

# ---------- start clean, so a failed earlier run doesn't interfere ----------
for p in (DB, SPARSE, DENSE):
    if os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)
os.makedirs(SPARSE)

# ---------- 1. features (C2) ----------
print("\n[1/5] Extracting features...")
pycolmap.extract_features(DB, IMAGES)

# ---------- 2. matching (C2) ----------
print("[2/5] Matching all images pairs...")
pycolmap.match_exhaustive(DB)

# ---------- 3. SfM: K, R, t + sparse points (C3 slide 39) ----------
print("[3/5] Running incremental SfM...")
maps = pycolmap.incremental_mapping(DB, IMAGES, SPARSE)
if len(maps) == 0:
    raise SystemExit("SfM failed: no reconstruction. Check overlap/sharpness of the photos.")

# COLMAP may split the scene into several pieces; keep the biggest one
best_id = max(maps, key=lambda k: maps[k].num_reg_images())
rec = maps[best_id]
print(f"Kept reconstruction {best_id}: "
      f"{rec.num_reg_images()} / {len(photos)} images registered, "
      f"{rec.num_points3D()} sparse points")
print(rec.summary())

# ---------- 4. undistort + resize (makes the pinhole model of C1a exact) ----------
print("[4/5] Undistorting images...")
opts = pycolmap.UndistortCameraOptions()
opts.max_image_size = MAX_IMAGE_SIZE
pycolmap.undistort_images(DENSE, os.path.join(SPARSE, str(best_id)), IMAGES,
                          undistort_options=opts)

# ---------- 5. save cameras as readable text ----------
print("[5/5] Writing text files...")
txt_dir = os.path.join(DENSE, "sparse_txt")
os.makedirs(txt_dir, exist_ok=True)
pycolmap.Reconstruction(os.path.join(DENSE, "sparse")).write_text(txt_dir)

print("\nDone.")
print("Undistorted images:", os.path.join(DENSE, "images"))
print("Camera text files: ", txt_dir)

# compute per-view depth maps using a depth estimation method of choice,
# but preferably one with little to no regularization (Scene reconstruction
# from high spatio-angular resolution light fields).