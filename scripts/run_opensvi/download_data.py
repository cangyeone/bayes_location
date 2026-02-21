import os
from googledrivedownloader import download_file_from_google_drive

PATH = "run_fm3d/run_opensvi/data"

# Ensure base dir exists
os.makedirs(PATH, exist_ok=True)

def dl(file_id: str, dest_path: str):
    download_file_from_google_drive(
        file_id=file_id,
        dest_path=dest_path,
        unzip=False,      # these are csv/pt/json; no unzip needed
        showsize=True,
        overwrite=False
    )

# Downloading velocity model .csv
#dl("1nUib66Psv2zoNnc4_EkNmVLTnHC6IkdU", f"{PATH}/Cahuilla_1D_VP.csv")
#dl("156aA_gNp_Tit0OutiM_VDv4fEwMTSqQX", f"{PATH}/Cahuilla_1D_VS.csv")

# Downloading travel-time models
#dl("17s3SxldoPiRB5HVVscrLsOwwG7jYcxtR", f"{PATH}/Cahuilla_1D_VP.pt")
#dl("1NrtlRKX2z66EFirqRINf6wztCbsHnU7K", f"{PATH}/Cahuilla_1D_VS.pt")

# Loading the station locations
dl("1qFecpeI3V0dwENd5Vj9_IoTATg8yLKXi", f"{PATH}/Cahuilla_Stations.csv")

# Downloading the Event Picks
#dl("12W7GLYtrkdkhFYxGfkaU_5hJbtpax9jN", f"{PATH}/Cahuilla_Picks.json")

# Downloading the pre-computed locations
#dl("1qw-FZ3oEVy7G5aUUBQ9IY3qJ7Gq56_U_", f"{PATH}/Cahuilla_Events.json")

# Ensure Events dir exists
os.makedirs(f"{PATH}/Events", exist_ok=True)
