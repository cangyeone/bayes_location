import re
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from obspy.geodetics import gps2dist_azimuth


# =========================
# 1. 读取第一个定位文件（你的合成 arrivals）
import pandas as pd
from pathlib import Path
import pandas as pd
from pathlib import Path

def read_catalog1(path):
    records = []
    path = Path(path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # ✅ 明确跳过首行
        header = next(f, None)
        print(header)
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 5:
                continue

            # event_id 在第 0 列
            eid_str = parts[0].split("/")[-1]
            eid_num = eid_str.split("_")[-1].split(".")[0]

            try:
                evid = int(eid_num)
                lat = float(parts[2])
                lon = float(parts[3])
                dep = float(parts[4])
            except ValueError:
                continue
            print(evid, lat, lon, dep)
            records.append(
                dict(id=evid, lat1=lat, lon1=lon, dep1=dep)
            )

    return pd.DataFrame(records)



# =========================
# 2. 读取第二个（REAL 输出：数字开头的是事件头/位置）
# =========================
_EVT_HEAD_RE = re.compile(r"^\s*(\d+)\s+\d{4}\s+\d{2}\s+\d{2}\s+\d{2}:\d{2}:\d{2}")

def _iter_real_text_files(p: Path):
    """
    给目录时：递归搜集可能的 REAL 文本输出文件。
    你也可以按你真实文件名规则在这里改过滤条件。
    """
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {p}")

    exts = {".txt", ".out", ".dat", ".real", ".loc", ".catalog", ".phase"}
    files = []
    for fp in p.rglob("*"):
        if fp.is_file():
            if fp.suffix.lower() in exts or fp.name.lower().startswith(("catalog", "real", "phase", "assoc", "loc")):
                files.append(fp)
    # 兜底：如果没匹配到后缀，也至少读目录下的普通文件（但避免超大二进制）
    if not files:
        for fp in p.rglob("*"):
            if fp.is_file() and fp.stat().st_size < 200 * 1024 * 1024:
                files.append(fp)
    return sorted(set(files))


def read_catalog2(path):
    """
    读取 REAL 输出（单文件或目录），抓取“数字开头的事件头行”：
    例子：
      0 2022 09 05 00:10:00.000 600.000 ... lat lon dep ...
      1 2022 09 05 00:10:01.000 610.000 ... lat lon dep ...
    """
    path = Path(path)
    files = _iter_real_text_files(path)

    records = []
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not _EVT_HEAD_RE.match(line):
                        continue
                    parts = line.split()
                    # 这里严格按你给的示例列位置
                    # [0]=id, [6]=lat, [7]=lon, [8]=dep
                    try:
                        evid = int(parts[0])
                        lat = float(parts[7])
                        lon = float(parts[8])
                        dep = float(parts[9])
                    except (ValueError, IndexError):
                        # 如果某些行列位不同/缺失，直接跳过
                        continue
                    #print(lat, lon, dep)
                    records.append(dict(id=evid, lat2=lat, lon2=lon, dep2=dep))
        except Exception:
            # 避免某个文件坏掉导致整个流程停
            continue

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No events parsed from REAL path: {path}")

    # 如果一个 id 出现多次（多文件/重复输出），保留最后一次或做平均
    # 这里用“最后一次”为准（也可以改成 groupby mean/median）
    df = df.sort_values(["id"]).drop_duplicates("id", keep="last").reset_index(drop=True)
    return df


# =========================
# 3. 计算误差
# =========================
def compute_errors(df):
    horiz = []
    for _, r in df.iterrows():
        dist_m, _, _ = gps2dist_azimuth(r.lat1, r.lon1, r.lat2, r.lon2)
        horiz.append(dist_m / 1000.0)
    df["horiz_err_km"] = horiz
    df["dep_err_km"] = df.dep2 - df.dep1
    return df


# =========================
# 4. 绘制地图
# =========================
def plot_map(df):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=proj)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="0.95")
    ax.gridlines(draw_labels=True)
    print(len(df), "events to plot.")
    ax.scatter(df.lon1, df.lat1, s=40, label="Catalog 1", transform=proj)
    ax.scatter(df.lon2, df.lat2, s=40, marker="x", label="Catalog 2 (REAL)", transform=proj)

    for _, r in df.iterrows():
        ax.plot([r.lon1, r.lon2], [r.lat1, r.lat2], color="0.5", linewidth=0.8, transform=proj)

    ax.legend()
    ax.set_title("Epicenter Comparison (Catalog 1 vs Catalog 2)")
    plt.show()


# =========================
# 5. 误差统计图
# =========================
def plot_errors(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df.horiz_err_km, bins=20)
    axes[0].set_xlabel("Horizontal error (km)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df.dep_err_km, bins=20)
    axes[1].set_xlabel("Depth error (km)")

    plt.tight_layout()
    plt.show()


# =========================
# main
# =========================
def main():
    cat1 = read_catalog1("run_fm3d/data/nlloc/all.locfiles.csv")

    # 这里传 REAL 输出目录或单文件都行
    # 如果你 REAL 的事件头就在某个具体文件里，也可以直接填那个文件路径
    cat2 = read_catalog2("run_fm3d/data/loc.synth_arrivals_skfmm_noise.new.txt")

    df = pd.merge(cat1, cat2, on="id", how="inner")
    df = compute_errors(df)

    print(df[["id", "horiz_err_km", "dep_err_km"]].describe())

    plot_map(df)
    plot_errors(df)


if __name__ == "__main__":
    main()
