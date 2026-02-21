import torch
import math
import torch.nn.functional as F
import torch.nn as nn 
import matplotlib.pyplot as plt
@torch.no_grad()
def _ensure_2d(a):
    if a.dim() == 1:
        return a[None, :]
    return a

def p_takeoff_angle_from_tt(model, x, xs, degrees=True, z_is_depth_positive=True):
    """
    model: PINNTravelTime, output (B,2) where [:,0]=Tp, [:,1]=Ts
    x  : (B,3) receiver [xr,yr,zr] in km
    xs : (B,3) source   [xe,ye,ze] in km
    return:
      angle: (B,) takeoff angle at source (from vertical down if z_is_depth_positive)
      dir  : (B,3) unit ray direction at source (pointing from source toward receiver along ray)
      grad : (B,3) dTp/dxs in s/km (slowness vector at source, up to sign convention)
    """
    x  = _ensure_2d(torch.as_tensor(x,  dtype=torch.float32, device=next(model.parameters()).device))
    xs = _ensure_2d(torch.as_tensor(xs, dtype=torch.float32, device=next(model.parameters()).device))

    # 需要对 xs 求梯度：不要 no_grad
    xs_req = xs.clone().detach().requires_grad_(True)

    out = model(x, xs_req)         # (B,2)
    Tp  = out[:, 0]                # (B,)

    # dTp/dxs: (B,3), 单位 s/km（因为 xs 是 km）
    grad = torch.autograd.grad(Tp.sum(), xs_req, create_graph=False, retain_graph=False)[0]

    # 射线方向（从震源指向台站的传播方向）取 -grad 的单位化
    eps = 1e-12
    dir_vec = -grad
    dir_unit = dir_vec / (dir_vec.norm(dim=-1, keepdim=True) + eps)

    # 竖直参考方向
    # 如果 z 是深度（向下为正），则 vertical_down = +z
    # 若 z 是高程（向上为正），则 vertical_down = -z
    vx, vy, vz = dir_unit[:, 0], dir_unit[:, 1], dir_unit[:, 2]
    if not z_is_depth_positive:
        vz = -vz  # 把“向下分量”统一到 vz

    # takeoff angle from vertical down: atan2(horizontal, down)
    horiz = torch.sqrt(vx*vx + vy*vy)
    print(horiz, vz)
    angle = torch.atan2(horiz, vz)  # vz<=0 会导致>90度；这里按“向下”为正的定义约束

    if degrees:
        angle = angle * (180.0 / math.pi)

    return angle.detach(), dir_unit.detach(), grad.detach()

class PINNTravelTimeTinny(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_merge = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus(), 
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr]
        # xs: (B,3) source   [xe,ye,ze]
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0 
        return out

class PINNTravelTime(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_merge = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus(), 
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr]
        # xs: (B,3) source   [xe,ye,ze]
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0
        return out

dtype = torch.float32 
device = torch.device('mps') 
travel_time = PINNTravelTimeTinny()
travel_time.eval()
travel_time.to(device) 
yy1 = []
ckpt_path = 'ckpt/tinny.v1.0.pt'
travel_time.load_state_dict(torch.load(ckpt_path, map_location='cpu')["model_state"])
travel_time.to(dtype)
# 单个事件-台站
xx = []

for i in range(300):
    xx.append(i)
    xr = torch.tensor([i,    100, 0.0], device=device)   # km
    xs = torch.tensor([100, 100, 30.0],  device=device)    # km (深度8km)
    angle_deg, dir_unit, grad = p_takeoff_angle_from_tt(travel_time, xr, xs, degrees=True)

    yy1.append(angle_deg.item())



travel_time = PINNTravelTime()
travel_time.eval()
travel_time.to(device) 
yy2 = []
ckpt_path = 'ckpt/time.v1.0.pt'
travel_time.load_state_dict(torch.load(ckpt_path, map_location='cpu')["model_state"])
travel_time.to(dtype)
# 单个事件-台站
xx = []

for i in range(300):
    xx.append(i)
    xr = torch.tensor([i,    100, 0.0], device=device)   # km
    xs = torch.tensor([100, 100, 30.0],  device=device)    # km (深度8km)
    angle_deg, dir_unit, grad = p_takeoff_angle_from_tt(travel_time, xr, xs, degrees=True)

    yy2.append(angle_deg.item())
plt.plot(xx, yy1, c="r", label="Rawdata v1.0")
plt.plot(xx, yy2, c="b", label="Eikonal v1.0")
plt.xlabel("Epicentral Distance (km)")
plt.ylabel("Takeoff Angle (deg)")
plt.legend()
plt.show()
print("takeoff angle (deg):", angle_deg.item())
print("unit ray direction:", dir_unit[0].cpu().numpy())
print("dTp/dxs (s/km):", grad[0].cpu().numpy())
