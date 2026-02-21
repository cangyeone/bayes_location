import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bisect import bisect_left, bisect_right
import datetime
import re
import numpy as np
btime1 = datetime.datetime(
                2022, 9, 5
            )
etime1 = datetime.datetime(
                2022, 9, 15
            )
def read_event_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    events = []
    locs = []
    i = 0
    for line in lines:
        # 读取事件头信息
        if line.startswith("#"):
            i = i + 1   
            header = line[1:].strip().split()
            # 时间
            try:
                origin_time = datetime.datetime.strptime(
                    " ".join(header[0:6]), "%Y %m %d %H %M %S.%f"
                )
            except Exception:
                origin_time = datetime.datetime.strptime(
                    " ".join(header[0:6]).replace("60", "59"),
                    "%Y %m %d %H %M %S.%f"
                )
            etime = origin_time 
            delta = (etime-btime1).total_seconds()
            lat = float(header[6])
            lon = float(header[7])
            depth = float(header[8])
            mag1 = float(header[9])
            mag2 = float(header[10])
            if origin_time < btime1 or origin_time > etime1:continue 
                #continue
            locs.append([lon, lat, depth, delta, 0, 0])
    #print("CCCC", i)
    return np.array(locs)


def read4(path):
    events = {}
    prov = []
    locs = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()

            year = int(parts[3])
            month = int(parts[4])
            day = int(parts[5])
            hour = int(parts[7])
            minute = int(parts[8])
            second = int(parts[9])
            microsecond = int(parts[10])

            etime = datetime.datetime(
                year, month, day,
                hour, minute, second,
                microsecond
            )
            if etime < btime1 or etime > etime1:
                continue
            try:
                i_loc = parts.index("LOC")
                i_dep = parts.index("DEP")
                i_mag = parts.index("MAG")
            except ValueError:
                return None
            try:
                lon = float(parts[i_loc + 1])
                lat = float(parts[i_loc + 2])
                depth_km = float(parts[i_dep + 1])
                mag_type = parts[i_mag + 1]
                mag = float(parts[i_mag + 2])
            except Exception:
                continue

            eloc = [lon, lat, depth_km]
            locs.append(eloc)

    return np.array(locs)


def read1(path, errors=[10, 20]):
    events = {}
    prov = []
    locs = []
    with open(path, "r") as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            #if "#" not in line:
            #    continue
            #line = line.split()
            header = line.strip().split(",")[:]
            # 时间
            #print(header)
            if len(header[1])==0:continue 
            etime = datetime.datetime.strptime(header[1], "%Y-%m-%dT%H:%M:%S.%f")
            if etime < btime1 or etime > etime1:
                continue
            lat = float(header[2])
            lon = float(header[3])
            depth = float(header[4])
            e1 = float(header[5])
            e2 = float(header[6])
            if e1 > errors[0] or e2 > errors[1]:
                continue
            #$etime = datetime.datetime.strptime(line[0] + " " + line[1], "%Y-%m-%d %H:%M:%S.%f")# - datetime.timedelta(hours=8)
            #if etime < begin or etime > end:
            #    continue

            eloc = [lon, lat, depth]
            locs.append(eloc)
            #events[tkey] = eloc
    #print(set(prov)) 
    return np.array(locs) 


#btime = datetime.datetime(2021, 5, 1, 0, 0)
#begin = datetime.datetime.strptime("2025-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
#end = datetime.datetime.strptime("2025-06-28 00:00:00", "%Y-%m-%d %H:%M:%S")
#btime1 = datetime.datetime.strptime(f"2021-05-21 00:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
#etime1 = datetime.datetime.strptime(f"2021-05-23 00:00:00.000", "%Y-%m-%d %H:%M:%S.%f")

def readctlg():
    file_ = open("data/2022.pha", "r", encoding="utf-8")
    p2i = {"Pg":0, "Sg":1, "Pn":2, "Sn":3}
    pdict = {"Pg":{}, "Sg":{}, "Pn":{}, "Sn":{}}
    #with open("odata/avialable.yangbi.oneday.txt", "r") as f:#读取可用台站和时段
    #    available = eval(f.read())
    mag = 0.0 

    events = {}    
    count = 0 
    #begin = datetime.datetime.strptime(f"2021-05-21 00:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
    #end = datetime.datetime.strptime(f"2021-05-23 00:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
    ndict = {"Pg":0, "Sg":0, "Pn":0, "Sn":0}
    events = []
    for line in file_.readlines():  
        if "NONE" in line:continue 
        if "#EVENT" in line:
            sline = line.strip().split()
            mag = float(sline[-1])
            etime = datetime.datetime.strptime(f"{sline[3]}-{sline[4]}-{sline[5]} {sline[7]}:{sline[8]}:{sline[9]}.{sline[10]}", "%Y-%m-%d %H:%M:%S.%f")
            elon, elat, edep = float(sline[12]), float(sline[13]), float(sline[15])
            #if sline[4] not in ["05"]:continue 
            #if sline[5] not in ["21", "22"]:continue 
            #@print(etime, elon, elat, edep, mag)
            if elat < 29.0 or elat > 30.4:continue 
            if elon < 101.8 or elon > 102.4: continue
            #if mag < 4:continue 
            delta = (etime-btime1).total_seconds()
            tkey = int(delta)
            count += 1  
            #72,72
            if etime < btime1 or etime > etime1:
               continue
            events.append([elon, elat, edep, delta, mag])
    
    return np.array(events) 

def read2(erange=[15, 30]):
    lons, lats, deps = [], [], []
    with open("run_fm3d/data/ours/reloc.real.True.txt", "r") as f:
        events = []
        for line in f:
            if line.startswith("#"):
                p = line[1:].strip().split(",")
                lon, lat, dep = map(float, p[2:5])
                e1, e2, e3 = map(float, p[11:11+3])  # 若需要可启用筛选
                eh = np.sqrt(e1**2 + e2**2)
                ez = e3 
                etime = datetime.datetime.strptime(p[1], "%Y-%m-%d %H:%M:%S.%f")
                delta = (etime-btime1).total_seconds()
                if eh > erange[0] or ez > erange[1]:
                    continue
                events.append([lon, lat, dep, delta])
    events = np.array(events)
    return events
# ---------------------------
# Utilities
# ---------------------------
EARTH_R_KM = 6371.0

def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km)."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return EARTH_R_KM * c

def add_panel_label(ax, label):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14, fontweight="bold"
    )

def mean_std_str(x):
    if len(x) == 0:
        return "--"
    return f"{np.mean(x):.2f} / {np.std(x, ddof=1) if len(x)>1 else 0.0:.2f}"

# ---------------------------
# Matching: reference (truth) vs method catalog
# TP 조건: |dt|<=dt_thr AND dh<=dh_thr
# Greedy one-to-one matching by best score within time window
# ---------------------------
def match_tp(reference, method_orig, dt_thr=3.0, dh_thr=30.0, time_window=10.0, err_thr=[15, 30]):
    """
    reference: [N, 5] -> lon, lat, dep, delta_sec, mag
    method:    [M, 4] -> lon, lat, dep, delta_sec, eh, ez
    returns:
      matched_idx_ref: list of ref indices
      matched_idx_m:   list of method indices
      errors: dict with arrays: dt, dh, dz
    """
    if len(reference) == 0 or len(method_orig) == 0:
        return [], [], {"dt": np.array([]), "dh": np.array([]), "dz": np.array([])}
    method = []
    for lon, lat, dep, delta_sec, eh, ez in method_orig:
        if eh <= err_thr[0] and ez <= err_thr[1]:
            method.append([lon, lat, dep, delta_sec, eh, ez])
    method = np.array(method)
    ref_t = reference[:, 3]
    met_t = method[:, 3]

    # sort method by time for fast window search
    met_order = np.argsort(met_t)
    met_t_sorted = met_t[met_order]

    used = np.zeros(len(method), dtype=bool)

    matched_ref = []
    matched_met = []
    dt_list, dh_list, dz_list = [], [], []

    for i in range(len(reference)):
        t0 = ref_t[i]

        # candidate indices in sorted-by-time array within +/- time_window
        lo = bisect_left(met_t_sorted, t0 - time_window)
        hi = bisect_right(met_t_sorted, t0 + time_window)
        if lo >= hi:
            continue

        cand_sorted_idx = met_order[lo:hi]
        # remove used
        cand_sorted_idx = cand_sorted_idx[~used[cand_sorted_idx]]
        if cand_sorted_idx.size == 0:
            continue

        # compute dt and dh for candidates
        dt = np.abs(method[cand_sorted_idx, 3] - t0)
        dh = haversine_km(
            reference[i, 0], reference[i, 1],
            method[cand_sorted_idx, 0], method[cand_sorted_idx, 1]
        )

        # enforce TP thresholds
        ok = (dt <= dt_thr) & (dh <= dh_thr)
        if not np.any(ok):
            continue

        cand_ok = cand_sorted_idx[ok]
        dt_ok = dt[ok]
        dh_ok = dh[ok]

        # choose best by normalized score (stable + explainable)
        score = (dt_ok / dt_thr)**2 + (dh_ok / dh_thr)**2
        j = cand_ok[np.argmin(score)]

        # mark used and record
        used[j] = True
        matched_ref.append(i)
        matched_met.append(j)

        dt_list.append(method[j, 3] - t0)  # signed (sec); use abs later if needed
        dh_list.append(haversine_km(reference[i,0], reference[i,1], method[j,0], method[j,1]))
        dz_list.append(method[j, 2] - reference[i, 2])  # signed depth diff (km)

    return matched_ref, matched_met, {
        "dt": np.array(dt_list, dtype=float),
        "dh": np.array(dh_list, dtype=float),
        "dz": np.array(dz_list, dtype=float),
    }
def match_tp_basic(reference, method, dt_thr=3.0, dh_thr=30.0, time_window=10.0):
    """
    reference: [N,5] lon,lat,dep,delta,mag
    method:    [M,>=4] lon,lat,dep,delta,(...)
    TP条件：|dt|<=dt_thr AND horizontal_dist<=dh_thr
    返回：tp_count, used_mask
    """
    if len(reference) == 0 or len(method) == 0:
        return 0, np.zeros(len(method), dtype=bool)

    ref_t = reference[:, 3]
    met_t = method[:, 3]

    order = np.argsort(met_t)
    met_t_sorted = met_t[order]
    used = np.zeros(len(method), dtype=bool)

    tp = 0
    for i in range(len(reference)):
        t0 = ref_t[i]
        lo = bisect_left(met_t_sorted, t0 - time_window)
        hi = bisect_right(met_t_sorted, t0 + time_window)
        if lo >= hi:
            continue

        cand = order[lo:hi]
        cand = cand[~used[cand]]
        if cand.size == 0:
            continue

        dt = np.abs(method[cand, 3] - t0)
        dh = haversine_km(reference[i,0], reference[i,1], method[cand,0], method[cand,1])

        ok = (dt <= dt_thr) & (dh <= dh_thr)
        if not np.any(ok):
            continue

        cand_ok = cand[ok]
        dt_ok = dt[ok]
        dh_ok = dh[ok]
        score = (dt_ok/dt_thr)**2 + (dh_ok/dh_thr)**2  # 可解释、稳定
        j = cand_ok[np.argmin(score)]

        used[j] = True
        tp += 1

    return tp, used
def prf1_vs_unc_threshold(reference, method_with_unc, T_list,
                          dt_thr=3.0, dh_thr=30.0, time_window=10.0,
                          vertical_ratio=2.0):
    """
    横轴 T：筛选阈值
    筛选：eh<=T 且 ez<=vertical_ratio*T
    然后在筛选后的集合上，使用 match_tp_basic 计算 TP
    Precision = TP / N_selected
    Recall    = TP / N_reference
    F1        = 2PR/(P+R)
    """
    n_ref = len(reference)
    P = np.zeros_like(T_list, dtype=float)
    R = np.zeros_like(T_list, dtype=float)
    F1 = np.zeros_like(T_list, dtype=float)

    if n_ref == 0:
        return P, R, F1

    eh = method_with_unc[:, 4]
    ez = method_with_unc[:, 5]

    for k, T in enumerate(T_list):
        sel = (eh <= T) & (ez <= vertical_ratio * T)
        msel = method_with_unc[sel]

        n_sel = len(msel)
        if n_sel == 0:
            P[k] = 0.0
            R[k] = 0.0
            F1[k] = 0.0
            continue

        tp, used = match_tp_basic(reference, msel, dt_thr=dt_thr, dh_thr=dh_thr, time_window=time_window)
        
        prec = tp / n_sel
        rec  = tp / n_ref
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        #print(n_sel, tp, n_ref, prec, rec)
        P[k], R[k], F1[k] = prec, rec, f1

    return P, R, F1

# ---------------------------
# Your readers (keep your existing ones)
# NOTE: we need:
#   ref (truth): lon, lat, dep, delta, mag   -> readctlg()
#   Catalog_1.pha: lon, lat, dep, delta      -> read_event_file()
#   NLLoc: lon, lat, dep, delta              -> read1() needs to return delta
#   Our: lon, lat, dep, delta                -> read2() needs to return delta
# ---------------------------
# IMPORTANT PATCH:
# your read1() currently returns [lon, lat, depth] only; for time-matching we need delta_sec too.
# Make a time-aware wrapper for NLLoc and Our outputs using the timestamp string already parsed.

def read_nlloc_timeaware(path, erange=[5, 10], btime1=None, etime1=None):
    locs = []
    with open(path, "r") as f:
        f.readline()  # skip header
        for line in f:
            header = line.strip().split(",")
            if len(header) < 7:
                continue
            if len(header[1]) == 0:
                continue
            etime = datetime.datetime.strptime(header[1], "%Y-%m-%dT%H:%M:%S.%f")
            if (btime1 is not None and etime < btime1) or (etime1 is not None and etime > etime1):
                continue

            lat = float(header[2])
            lon = float(header[3])
            depth = float(header[4])
            e1 = float(header[5])
            e2 = float(header[6])
            if e1 > erange[0] or e2 > erange[1]:
                continue

            delta = (etime - btime1).total_seconds()
            locs.append([lon, lat, depth, delta, e1, e2])
    return np.array(locs, dtype=float)

def read_ours_timeaware(path, erange=[15, 30], btime1=None, etime1=None):
    events = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            p = line[1:].strip().split(",")
            if len(p) < 14:
                continue

            lon, lat, dep = map(float, p[2:5])
            # your file: e1,e2,e3 at p[11:14] (as you used before)
            e1, e2, e3 = map(float, p[11:14])
            eh = np.sqrt(e1**2 + e2**2)
            ez = e3

            etime = datetime.datetime.strptime(p[1], "%Y-%m-%d %H:%M:%S.%f")
            if (btime1 is not None and etime < btime1) or (etime1 is not None and etime > etime1):
                continue
            if eh > erange[0] or ez > erange[1]:
                continue

            delta = (etime - btime1).total_seconds()
            events.append([lon, lat, dep, delta, eh, ez])
    return np.array(events, dtype=float)

import numpy as np
import datetime
import re

def _is_event_header(line: str) -> bool:
    # 事件头通常以 event_id 数字开头（你样例是 "1 2022 09 05 ...")
    s = line.strip()
    return bool(s) and s[0].isdigit()

def _parse_event_time_from_header(tokens):
    """
    tokens example:
    [id, YYYY, MM, DD, HH:MM:SS.sss, ...]
    """
    if len(tokens) < 5:
        return None
    yyyy, mm, dd = map(int, tokens[1:4])
    tstr = tokens[4]
    # 兼容秒带小数 / 不带小数
    if "." in tstr:
        fmt = "%H:%M:%S.%f"
    else:
        fmt = "%H:%M:%S"
    tt = datetime.datetime.strptime(tstr, fmt).time()
    return datetime.datetime(yyyy, mm, dd, tt.hour, tt.minute, tt.second, tt.microsecond)

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def read_catalog_block_style(
    path,
    erange=(15.0, 30.0),
    btime1=None,
    etime1=None,
    latlondep_cols=(7, 8, 9),  # 0-based indices for (lat, lon, dep) in header tokens
    return_picks=False,
):
    """
    Read block-style catalog:
      header line: starts with digit
      following lines: picks (ignored unless return_picks=True)

    Returns:
      events: np.ndarray of shape [Ne, 6] with columns [lon, lat, dep, delta, eh, ez]
      optionally picks_list: list of picks for each event
    """
    events = []
    all_picks = []

    cur = None  # dict for current event
    cur_picks = []

    def _flush_current():
        nonlocal cur, cur_picks
        if cur is None:
            return

        etime = cur.get("etime", None)
        if etime is None:
            cur = None
            cur_picks = []
            return

        # time window
        if (btime1 is not None and etime < btime1) or (etime1 is not None and etime > etime1):
            cur = None
            cur_picks = []
            return

        # delta requires btime1
        if btime1 is None:
            delta = np.nan
        else:
            delta = (etime - btime1).total_seconds()

        lon = cur["lon"]
        lat = cur["lat"]
        dep = cur["dep"]

        # NOTE: eh/ez unknown in your sample header -> keep NaN by default
        eh = cur.get("eh", -1)
        ez = cur.get("ez", -1)

        # error filter only if not NaN
        if np.isfinite(eh) and eh > float(erange[0]):
            cur = None
            cur_picks = []
            return
        if np.isfinite(ez) and ez > float(erange[1]):
            cur = None
            cur_picks = []
            return
        print("event:", lon, lat, dep, delta, eh, ez)
        events.append([lon, lat, dep, delta, eh, ez])
        if return_picks:
            all_picks.append(cur_picks)

        cur = None
        cur_picks = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            if _is_event_header(line):
                # new event begins -> flush previous
                _flush_current()

                tok = line.split()
                etime = _parse_event_time_from_header(tok)

                # parse lat lon dep
                ilat, ilon, idep = latlondep_cols
                lat = _safe_float(tok[ilat])
                lon = _safe_float(tok[ilon])
                dep = _safe_float(tok[idep])

                cur = dict(etime=etime, lon=lon, lat=lat, dep=dep)

                # OPTIONAL: try to auto-detect EH/EZ if the header contains keywords (rare)
                # If your real file has something like "... ERH 1.2 ERZ 3.4 ..."
                # this block can be extended once you show one real header line with EH/EZ.
                cur_picks = []
            else:
                # pick line (e.g. "SC  JLO  P  12647.7700 ...")
                if cur is None:
                    continue
                if return_picks:
                    cur_picks.append(line)

    # flush last event
    _flush_current()

    if return_picks:
        return np.array(events, dtype=float), all_picks
    return np.array(events, dtype=float)

import numpy as np
import datetime as dt

def read_catalog_hash_block_style(
    path,
    erange=(15.0, 30.0),          # (eh_max_km, ez_max_km)
    btime1=None,
    etime1=None,
    return_picks=False,
    parse_errors_from_tail=True,  # True: 用 header 末尾三个数当作 e1,e2,e3
):
    """
    Read '#'-header block-style catalog.

    Format:
      Header line:
        # YYYY MM DD HH MM SS.sss  LAT  LON  DEP  MAG  e1 e2 e3  ...
      Following lines (until next '#'):
        STA  T  W  PHASE
        e.g. "SCJLO 14.0500 1.0 P"

    Returns:
      events: np.ndarray [Ne,6] columns [lon, lat, dep, delta, eh, ez]
      optionally picks_list: list (len=Ne) of pick records for each event:
        each pick is (sta:str, t:float, w:float, phase:str)
    """

    events = []
    all_picks = []

    cur = None
    cur_picks = []

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    import datetime as dt

    def _parse_header_time(tok):
        # tok: ['#','YYYY','MM','DD','HH','MM','SS.sss', ...]
        y  = int(tok[1]); mo = int(tok[2]); d  = int(tok[3])
        hh = int(tok[4]); mm = int(tok[5])
        sec = float(tok[6])

        # 构造到分钟，然后用 timedelta 加秒数（允许 sec>=60 自动进位）
        base = dt.datetime(y, mo, d, hh, mm, 0, 0)
        return base + dt.timedelta(seconds=sec)

    def _flush_current():
        nonlocal cur, cur_picks
        if cur is None:
            return

        etime = cur["etime"]

        # time window
        if (btime1 is not None and etime < btime1) or (etime1 is not None and etime > etime1):
            cur = None
            cur_picks = []
            return

        # delta relative to btime1 if provided
        delta = (etime - btime1).total_seconds() if btime1 is not None else np.nan

        lon = cur["lon"]
        lat = cur["lat"]
        dep = cur["dep"]
        eh  = cur.get("eh", np.nan)
        ez  = cur.get("ez", np.nan)

        # error filter (only if finite)
        if np.isfinite(eh) and eh > float(erange[0]):
            cur = None
            cur_picks = []
            return
        if np.isfinite(ez) and ez > float(erange[1]):
            cur = None
            cur_picks = []
            return

        events.append([lon, lat, dep, delta, eh, ez])
        if return_picks:
            all_picks.append(cur_picks)

        cur = None
        cur_picks = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                # new event begins
                _flush_current()

                tok = line.split()
                # 基本健壮性检查
                if len(tok) < 10:
                    cur = None
                    cur_picks = []
                    continue

                etime = _parse_header_time(tok)

                # 按你示例的固定位置解析
                lat = _safe_float(tok[7])
                lon = _safe_float(tok[8])
                dep = _safe_float(tok[9])

                cur = dict(etime=etime, lon=lon, lat=lat, dep=dep)

                # 解析误差：默认用 header 的 e1,e2,e3 = tok[11], tok[12], tok[13]
                if parse_errors_from_tail and len(tok) >= 14:
                    e1 = _safe_float(tok[11])
                    e2 = _safe_float(tok[12])
                    e3 = _safe_float(tok[13])
                    if np.isfinite(e1) and np.isfinite(e2):
                        cur["eh"] = float(np.hypot(e1, e2))
                    else:
                        cur["eh"] = np.nan
                    cur["ez"] = float(e3) if np.isfinite(e3) else np.nan
                else:
                    cur["eh"] = np.nan
                    cur["ez"] = np.nan

                cur_picks = []
            else:
                # pick line
                if cur is None:
                    continue

                if return_picks:
                    # e.g. "SCJLO    14.0500 1.0 P"
                    tok = line.split()
                    if len(tok) < 4:
                        continue
                    sta = tok[0]
                    t = _safe_float(tok[1])
                    w = _safe_float(tok[2])
                    ph = tok[3]
                    cur_picks.append((sta, float(t), float(w), ph))

    _flush_current()

    events = np.asarray(events, dtype=float)
    if return_picks:
        return events, all_picks
    return events

def match_tp_with_dz(reference, method, dt_thr=3.0, dh_thr=30.0, dz_thr=60.0, time_window=10.0):
    """
    TP条件：|dt|<=dt_thr AND dh<=dh_thr AND |dz|<=dz_thr
    reference: [N,5] lon,lat,dep,delta,mag
    method:    [M,4] lon,lat,dep,delta
    return: matched indices + errors + used mask (for FP)
    """
    if len(reference) == 0 or len(method) == 0:
        return [], [], {"dt": np.array([]), "dh": np.array([]), "dz": np.array([])}, np.zeros(len(method), dtype=bool)

    ref_t = reference[:, 3]
    met_t = method[:, 3]

    met_order = np.argsort(met_t)
    met_t_sorted = met_t[met_order]
    used = np.zeros(len(method), dtype=bool)

    matched_ref, matched_met = [], []
    dt_list, dh_list, dz_list = [], [], []

    for i in range(len(reference)):
        t0 = ref_t[i]

        lo = bisect_left(met_t_sorted, t0 - time_window)
        hi = bisect_right(met_t_sorted, t0 + time_window)
        if lo >= hi:
            continue

        cand = met_order[lo:hi]
        cand = cand[~used[cand]]
        if cand.size == 0:
            continue

        dt = np.abs(method[cand, 3] - t0)
        dh = haversine_km(reference[i, 0], reference[i, 1], method[cand, 0], method[cand, 1])
        dz = np.abs(method[cand, 2] - reference[i, 2])

        ok = (dt <= dt_thr) & (dh <= dh_thr) & (dz <= dz_thr)
        if not np.any(ok):
            continue

        cand_ok = cand[ok]
        dt_ok, dh_ok, dz_ok = dt[ok], dh[ok], dz[ok]

        # 归一化 score：时间、水平、垂直一起约束，避免“只靠时间匹配”
        score = (dt_ok / dt_thr)**2 + (dh_ok / dh_thr)**2 + (dz_ok / dz_thr)**2
        j = cand_ok[np.argmin(score)]

        used[j] = True
        matched_ref.append(i)
        matched_met.append(j)

        dt_list.append(method[j, 3] - t0)                  # signed
        dh_list.append(haversine_km(reference[i,0], reference[i,1], method[j,0], method[j,1]))
        dz_list.append(method[j, 2] - reference[i, 2])      # signed

    return matched_ref, matched_met, {
        "dt": np.array(dt_list, dtype=float),
        "dh": np.array(dh_list, dtype=float),
        "dz": np.array(dz_list, dtype=float),
    }, used
def pr_curve_vs_threshold(reference, method, dt_thr=3.0, r_list=None, time_window=10.0):
    """
    横轴 r：水平阈值 (km)
    垂直阈值 = 2r
    时间阈值 dt_thr 固定
    返回：r, precision, recall
    """
    if r_list is None:
        r_list = np.arange(0, 101, 5)  # 0..100 km, step=5

    prec = np.zeros_like(r_list, dtype=float)
    rec  = np.zeros_like(r_list, dtype=float)

    n_ref = len(reference)
    n_met = len(method)

    for k, r in enumerate(r_list):
        dz_thr = 2.0 * float(r)

        mref, mmet, err, used = match_tp_with_dz(
            reference, method,
            dt_thr=dt_thr,
            dh_thr=float(r),
            dz_thr=dz_thr,
            time_window=time_window
        )

        tp = len(mref)
        fn = n_ref - tp
        fp = n_met - int(np.sum(used))   # 未被匹配使用的 method events

        prec[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[k]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return r_list, prec, rec

# ---------------------------
# Main: compute TP + plot + latex table
# ---------------------------
def make_error_benchmark_plots_and_table(
    btime1, etime1,
    dt_thr=3.0, dh_thr=30.0,
    #locs_a=None, locs_b=None, locs_c=None, locs_d=None,
    out_fig="figures/figs/fig_error_compare.png",
    out_fig_pdf="figures/figs/fig_error_compare.pdf"
):
    # ---- Read reference (truth): lon lat dep delta mag
    ref = readctlg()  # must return lon,lat,dep,delta,mag
    if ref.shape[1] != 5:
        raise ValueError("readctlg() must return [lon, lat, dep, delta_sec, mag].")
    
    # ---- Read method catalogs (time-aware)
    def abs_err_pack(err_dict):
        dh = np.abs(err_dict["dh"])
        dz = np.abs(err_dict["dz"])
        dt = np.abs(err_dict["dt"])
        return dh, dz, dt
    def stats_row(name, recall, dh, dz, dt, num=0):
        return (
            name,
            f"{recall:.3f}",
            mean_std_str(dh),
            mean_std_str(dz),
            mean_std_str(dt),
            f"{num:3d}", 
        )    

    erange = [15, 30]
    dt_thr = 3.0 
    dh_thr = 20.0
    file_names = [
        #"run_fm3d/data/ours/reloc.real.True.v1.0.txt",
        "run_fm3d/data/ours/reloc.crop.True.v1.0.txt",
        "run_fm3d/data/ours/reloc.real.True.eikonal.v1.0.txt",
        "run_fm3d/data/ours/reloc.real.realtime.True.v1.0.txt",
        "run_fm3d/data/nlloc_sc_crop/all.locfiles.csv", 
        #"run_fm3d/data/ours/reloc.real.True.withoutele.v1.0.txt",
        
        #"run_fm3d/data/nlloc_sc/all.locfiles.csv",
        #"data/Catalog_1.Pha"
    ]
    read_method = [
        #read_ours_timeaware, 
        read_ours_timeaware,
        read_ours_timeaware,
        read_ours_timeaware,
        read_nlloc_timeaware,
        #read_ours_timeaware,
        #read_catalog_hash_block_style, 
    ]
    names = [
        #"Ours",
        "Ours",
        "Ours with eikonal",
        "Ours with real data",
        "NonLinLoc",
    ]
    eranges = [
        [10, 20], 
        [13, 26],
        [20, 40],
        [3.2, 6.4],
    ]
    rows_all = []
    rows_ge3 = []
    for fnamn, readm, nms, erange in zip(file_names, read_method, names, eranges):
        cat = readm(fnamn, erange=erange, btime1=btime1, etime1=etime1)
        m_ref, m_cat, err = match_tp(ref, cat, dt_thr=dt_thr, dh_thr=dh_thr, err_thr=erange)
    

        # TP sample masks in reference (for M>=3 stats we use matched reference magnitude)
        mag_ref = ref[:, 4]
        idx_ref = np.array(m_ref, dtype=int)
   
        total_ref_all = len(ref)
        recall_all = len(idx_ref) / total_ref_all if total_ref_all else 0.0
    
        ref_ge3 = mag_ref >= 3.0
        total_ref_ge3 = int(np.sum(ref_ge3))
        recall_ge3 = np.sum(ref_ge3[idx_ref]) / total_ref_ge3 if total_ref_ge3 else 0.0
       # ---- Build arrays for stats (use absolute errors for reporting)


        dh, dz, dt = abs_err_pack(err)

        # ---- Subset for M>=3 using matched reference index
        def subset_by_mag_ge3(dh, dz, dt, idx_ref):
            if len(idx_ref) == 0:
                return np.array([]), np.array([]), np.array([])
            m = mag_ref[idx_ref] >= 3.0
            return dh[m], dz[m], dt[m]

        dh_ge3, dz_ge3, dt_ge3 = subset_by_mag_ge3(dh, dz, dt, idx_ref)
        rows_all.append(stats_row(nms, recall_all, dh, dz, dt, len(cat)))
        rows_ge3.append(stats_row(nms, recall_ge3, dh_ge3, dz_ge3, dt_ge3))


    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance comparison against the reference (manual) catalog using TP matches defined by $|\Delta t|\leq 3$~s and horizontal distance error $\leq 30$~km. Reported errors are absolute values; mean/std are shown as mean / std.}")
    latex.append(r"\label{tab:eikonal_benchmark}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Recall & Horizontal error (km) & Depth error (km) & Time error (s) & Num.\\")
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{5}{l}{\textbf{All events}}\\")
    for r in rows_all:
        latex.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} \\\\")
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{5}{l}{\textbf{Events with $M \ge 3$}}\\")
    for r in rows_ge3:
        latex.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)

# ---------------------------
# Run
# ---------------------------
# you already have btime1, etime1 in your script
btime1 = datetime.datetime(
                2022, 9, 5
            )
etime1 = datetime.datetime(
                2022, 9, 15
            )
result = make_error_benchmark_plots_and_table(
    btime1=btime1, etime1=etime1,
    dt_thr=3.0, dh_thr=30.0,
    
    out_fig="figures/figs/fig_eikonal_compare.png",
    out_fig_pdf="figures/figs/fig_eikonal_compare.pdf"
)


