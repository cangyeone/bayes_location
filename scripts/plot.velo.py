import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

meta_json_path = Path("run_fm3d/data/xyz_vp_vs_meta.json")  # 改成你的json路径

meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
axes_path = Path(meta["axes_file"])
npy_path  = Path(meta["npy_file"])

axes = np.load(axes_path)
x_axis = axes["x_km"]
y_axis = axes["y_km"]
z_axis = axes["z_km"]

arr = np.load(npy_path)  # (N,5) columns [x,y,z,vp,vs]

Nz, Ny, Nx = meta["shape"]
assert (Nz, Ny, Nx) == (len(z_axis), len(y_axis), len(x_axis)), "shape mismatch!"

# 关键：因为写出顺序是 z 外层、y 中层、x 内层，所以 reshape 为 (Nz,Ny,Nx)
VP = arr[:,3].reshape(Nz, Ny, Nx)
VS = arr[:,4].reshape(Nz, Ny, Nx)
print(VP.shape, VS.shape)
def nearest_index(axis, value):
    return int(np.argmin(np.abs(axis - value)))

iz0   = nearest_index(z_axis, 0.0)
izmid = nearest_index(z_axis, 0.5*(z_axis[0]+z_axis[-1]))
iy0   = nearest_index(y_axis, 0.0)
ix0   = nearest_index(x_axis, 0.0)

# 1) Vp at z=0
plt.figure()
plt.imshow(VP[iz0,:,:], origin="lower",
           extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
           aspect="auto")
plt.colorbar(label="Vp (km/s)")
plt.title(f"Vp at z={z_axis[iz0]:.1f} km")
plt.xlabel("x (km)"); plt.ylabel("y (km)")
plt.tight_layout(); plt.show()

# 2) Vs at mid z
plt.figure()
plt.imshow(VS[izmid,:,:], origin="lower",
           extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
           aspect="auto")
plt.colorbar(label="Vs (km/s)")
plt.title(f"Vs at z={z_axis[izmid]:.1f} km")
plt.xlabel("x (km)"); plt.ylabel("y (km)")
plt.tight_layout(); plt.show()

# 3) Vp section y=0 (x-z)
plt.figure()
plt.imshow(VP[:,iy0,:], origin="upper",
           extent=[x_axis[0], x_axis[-1], z_axis[-1], z_axis[0]],
           aspect="auto")
plt.colorbar(label="Vp (km/s)")
plt.title(f"Vp section at y={y_axis[iy0]:.1f} km")
plt.xlabel("x (km)"); plt.ylabel("z (km, down)")
plt.tight_layout(); plt.show()

# 4) Vs section x=0 (y-z)
plt.figure()
plt.imshow(VS[:,:,ix0], origin="upper",
           extent=[y_axis[0], y_axis[-1], z_axis[-1], z_axis[0]],
           aspect="auto")
plt.colorbar(label="Vs (km/s)")
plt.title(f"Vs section at x={x_axis[ix0]:.1f} km")
plt.xlabel("y (km)"); plt.ylabel("z (km, down)")
plt.tight_layout(); plt.show()

print("Vp range:", float(VP.min()), float(VP.max()))
print("Vs range:", float(VS.min()), float(VS.max()))
