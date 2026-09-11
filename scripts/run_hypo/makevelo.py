import numpy as np

def haversine_km(lon1, lat1, lon2, lat2):
    # great-circle distance (km)
    R = 6371.0
    lon1 = np.deg2rad(lon1); lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2); lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0*np.arcsin(np.sqrt(a))
    return R * c

def layer_mean(values, weights=None, mean_type="arith"):
    if weights is None:
        if mean_type == "arith":
            return np.mean(values)
        elif mean_type == "slowness":
            return 1.0 / np.mean(1.0 / values)
        else:
            raise ValueError("mean_type must be 'arith' or 'slowness'")
    else:
        w = np.asarray(weights, dtype=float)
        wsum = np.sum(w)
        if wsum <= 0:
            return np.nan
        if mean_type == "arith":
            return np.sum(w * values) / wsum
        elif mean_type == "slowness":
            return 1.0 / (np.sum(w * (1.0 / values)) / wsum)
        else:
            raise ValueError("mean_type must be 'arith' or 'slowness'")

def convert_3d_to_1d(
    arr,                       # (N,5) -> lon,lat,dep,Vp,Vs
    mode="area",               # "area" or "point"
    mean_type="arith",         # "arith" or "slowness"
    lon0=None, lat0=None,      # for point mode
    sigma_km=30.0              # for point mode: gaussian width
):
    lon = arr[:,0]; lat = arr[:,1]; dep = arr[:,2]
    vp  = arr[:,3]; vs  = arr[:,4]

    deps = np.unique(dep)
    deps.sort()

    out = np.zeros((deps.size, 3), dtype=float)  # dep, Vp1D, Vs1D
    for i, z in enumerate(deps):
        m = (dep == z)
        vp_z = vp[m]
        vs_z = vs[m]

        if mode == "area":
            w = None
        elif mode == "point":
            if lon0 is None or lat0 is None:
                raise ValueError("point mode requires lon0, lat0")
            d = haversine_km(lon[m], lat[m], lon0, lat0)
            w = np.exp(-0.5 * (d / sigma_km)**2)
        else:
            raise ValueError("mode must be 'area' or 'point'")

        vp1d = layer_mean(vp_z, weights=w, mean_type=mean_type)
        vs1d = layer_mean(vs_z, weights=w, mean_type=mean_type)
        out[i,:] = (z, vp1d, vs1d)

    return out  # columns: dep, Vp1D, Vs1D

if __name__ == "__main__":
    # 1) load from file (comma-separated)
    # e.g., model.csv lines: lon,lat,dep_km,Vp,Vs
    arr = np.loadtxt("run_fm3d/data/velo.txt", delimiter=",")

    # A) 区域 1D（每层横向算术平均）
    prof_area = convert_3d_to_1d(arr, mode="area", mean_type="arith")
    np.savetxt("model_1d_area.txt", prof_area, fmt="%.3f", header="dep_km Vp_km_s Vs_km_s")

    # B) 某点附近 1D（例如 lon0=97.2, lat0=21.0，30 km 高斯加权；用慢度平均）
    prof_point = convert_3d_to_1d(
        arr, mode="point", mean_type="slowness",
        lon0=97.2, lat0=21.0, sigma_km=30.0
    )
    np.savetxt("model_1d_point.txt", prof_point, fmt="%.3f", header="dep_km Vp_km_s Vs_km_s")
