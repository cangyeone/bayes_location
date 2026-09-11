from HypoSVI import location as lc

import torch
import torch.nn as nn
import pandas as pd 

class EikoNetIOWrapper(nn.Module):
    """
    Wrap a trained model (single phase: P or S) to match EikoNet-like I/O:
      - Input Xp: (B,6) = [xs, ys, zs, xr, yr, zr]
      - TravelTimes(Xp) -> (B,) travel time (seconds)
      - Velocity(Xp)    -> (B,) speed from eikonal gradient: v = 1 / ||∇_xr T||

    Notes:
    - Assumes underlying model forward(xr, xs) returns travel time in seconds with shape (B,) or (B,1) or (B,2).
      If it returns (B,2), we will select channel by `phase` (0 for P, 1 for S).
    - projection/normlisation arguments are accepted for signature compatibility but not applied by default.
    """
    def __init__(self, trained_model: nn.Module, phase: str, device, channel_map=None):
        super().__init__()
        self.model = trained_model.to(device)
        self.model.eval()
        self.device = device
        self.phase = phase.upper()

        # If your underlying model ever returns 2 channels, this decides which to pick.
        # You said P/S are different models; so this likely won’t be used, but it’s harmless.
        if channel_map is None:
            channel_map = {"P": 0, "S": 1}
        self.channel_map = channel_map

    def _split_Xp(self, Xp):
        Xp = Xp.to(self.device)
        xs = Xp[:, 0:3]
        xr = Xp[:, 3:6]
        return xr, xs

    def _select_T(self, out):
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        if out.ndim == 1:
            return out
        if out.ndim == 2 and out.shape[1] >= 2:
            return out[:, self.channel_map.get(self.phase, 0)]
        raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")

    def TravelTimes(self, Xp, projection=False, normlisation=False):
        # 注意：这里不要 no_grad
        xr, xs = self._split_Xp(Xp)
        out = self.model(xr, xs)
        return self._select_T(out)


    def Velocity(self, Xp: torch.Tensor, projection=False, normlisation=False):
        xr, xs = self._split_Xp(Xp)
        xr = xr.clone().detach().requires_grad_(True)
        xs = xs.clone().detach()

        out = self.model(xr, xs)
        T = self._select_T(out)  # (B,)

        g = torch.autograd.grad(
            outputs=T.sum(),
            inputs=xr,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # (B,3)

        v = 1.0 / (torch.linalg.norm(g, dim=-1) + 1e-12)
        return v

    def forward(self, Xp):
        return self.TravelTimes(Xp)
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
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr]  (km)
        # xs: (B,3) source   [xe,ye,ze]  (km)
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out.sigmoid() * 1000.0  # seconds
        return out

dtype = torch.float32 
device = torch.device('mps') 
travel_time = PINNTravelTime()
travel_time.eval()
travel_time.to(device) 

ckpt_path = 'ckpt/time.v3.pt'
travel_time.load_state_dict(torch.load(ckpt_path, map_location='cpu')["model_state"])
travel_time.to(dtype)
model_p = EikoNetIOWrapper(travel_time, device=device, phase='P')
model_s = EikoNetIOWrapper(travel_time, device=device, phase='S') 

# Loading the full phase picking for 100 events
EVT  = lc.IO_JSON("run_fm3d/run_opensvi/data/Cahuilla_Picks.json",rw_type='r')
# The event observations are split in a dictionary format where each key is given by the event ID
event_ids = EVT.keys()
print('Event Ids:{}'.format(event_ids))
EVT_new = {}
count = 0 
for evt_id, val in EVT.items():
    EVT_new[evt_id] = val 
    count += 1 
    if count > 10:
        break 
# Inspecting one of these events we see there is a sub-directory that has picks, representing a pandas dataframe with
# observational pick times, pick error and phase type.

Stations       = pd.read_csv("run_fm3d/run_opensvi/data/Cahuilla_Stations.csv",names=['Network','Station','Y','X','Elevation'],sep=r'\s+')
Stations       = Stations.drop_duplicates(['Network','Station'],keep='last').reset_index(drop=True)
Stations['Z']  = -(Stations['Elevation']/1000)

LocMethod = lc.HypoSVI([model_p, model_s],Phases=['P','S'], device=device)
LocMethod.LocateEvents(EVT_new,Stations,output_plots=True, output_path="run_fm3d/run_opensvi/data/Events", timer=True)