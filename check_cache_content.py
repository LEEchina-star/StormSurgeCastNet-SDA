#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

# ===================== Your real import =====================
from train_mps import coastalLoader

# ===================== Your real data path =====================
root = "/Volumes/code_copy/科研工作2026/StormSurgeCastNet-main/Data2"

# ===================== Load split and statistics =====================
print("Loading splits and statistics...")
splits_ids     = np.load(root + "/aux/splits_ids.npy", allow_pickle=True).item()
stats_data     = np.load(root + "/aux/stats.npy", allow_pickle=True).item()
stats_ibtracs  = np.load(root + "/aux/stats_ibtracs.npy", allow_pickle=True).item()

# ===================== Config (same as training) =====================
class Config:
    input_t = 12
    drop_data = 0.25
    context = 128
    res = 0.025
    center_gauge = True
    no_gesla_context = False

config = Config()
train_lead = 0

# ===================== Create dataset =====================
print("Creating dataset...")
dataset = coastalLoader(
    root=root,
    split='train',
    hyperlocal=True,
    splits_ids=splits_ids,
    stats=stats_data,
    stats_ibtracs=stats_ibtracs,
    input_len=config.input_t,
    drop_in=config.drop_data,
    context_window=config.context,
    res=config.res,
    lead_time=train_lead,
    center_gauge=config.center_gauge,
    no_gesla_context=config.no_gesla_context
)

# ===================== Find first valid sample =====================
sample = None
for i in range(len(dataset)):
    sample = dataset[i]
    if sample is not None:
        print(f"✅ Found valid sample at index = {i}")
        break

if sample is None:
    print("❌ No valid samples found!")
    exit()

# ===================== Extract key data =====================
inputs = sample["input"]
era5     = inputs["era5"]
sparse   = inputs["sparse"]
series   = inputs["series"]

print("\n===== Sample Shapes =====")
print("era5      :", era5.shape)
print("sparse    :", sparse.shape)
print("series    :", series.shape)
#%%
# ===================== Plot (ALL ENGLISH) =====================
plt.rcParams['figure.figsize'] = (16, 12)

# 1. ERA5 MSL Pressure
plt.subplot(2,2,1)
plt.imshow(era5[0, 0], cmap='jet')
plt.title("ERA5 Mean Sea Level Pressure (Hour 0)", fontsize=14)
plt.colorbar()

# 2. ERA5 U10 Wind
plt.subplot(2,2,2)
plt.imshow(era5[0, 1], cmap='RdBu_r')
plt.title("ERA5 U10 Wind (Hour 0)", fontsize=14)
plt.colorbar()

# 3. GESLA Sparse Water Level Grid
plt.subplot(2,2,3)
plt.imshow(sparse[10, 0], cmap='ocean')
plt.title("GESLA Storm Surge Grid (Hour 0)", fontsize=14)
plt.colorbar()

# 4. Center Gauge Time Series
plt.subplot(2,2,4)
plt.plot(range(12), series[:, 0], 'b-o', linewidth=3, markersize=8)
plt.grid(True)
plt.title("Central Gauge: 12-hour Input Sequence", fontsize=14)
plt.xlabel("Hour")
plt.ylabel("Normalized Storm Surge")

plt.tight_layout()
plt.savefig("model_input_sample.png", dpi=200)
plt.show()
# %%
