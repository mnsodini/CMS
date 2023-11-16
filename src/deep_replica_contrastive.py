import h5py
import os
# Limit display of TF messages to errors only 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")


# workdir = "/work/submit/deep1018/a3d3-hackathon-2023/"
features_dataset = np.load('../data/datasets_-1.npz')


from nflows.nn.nets import ResidualNet
from nflows import transforms, distributions, flows

print("Loading latent space data")
batch_size = 4000
# test_batch_size = 10000
test_batch_size = 80
latent_size = 6 

data = np.load('Data/model_latent.npz')
val_data = torch.from_numpy(data['x_val'].reshape(-1, latent_size)).to(dtype=torch.float32, device=device)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,)
test_data = torch.from_numpy(data['x_test'].reshape(-1, latent_size)).to(dtype=torch.float32, device=device)
test_data_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True,)

    
### NORMLAIZING FLOW 

num_layers = 4
base_dist = distributions.StandardNormal(shape=[latent_size])

transform_layers = []
for _ in range(num_layers):
    transform_layers.append(transforms.permutations.ReversePermutation(features=latent_size))
    transform_layers.append(
        transforms.autoregressive.MaskedAffineAutoregressiveTransform(
            features=latent_size,
            hidden_features=5
        )
    )

transform_obj = transforms.CompositeTransform(transform_layers)

flow_obj = flows.Flow(transform_obj, base_dist).to(device)

optimizer = torch.optim.Adam(flow_obj.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# num_iter = 10
# for i in range(num_iter):
#     # use the val_data_loader from previous step; this is only for quicker training
#     # alternatively restrict to fewer batches from train data loader
#     for idx, val in enumerate(val_data_loader):
#         # to work with the reshaping below
#         if val.shape[0] != batch_size:
#             continue
#         optimizer.zero_grad()
#         loss = -flow_obj.log_prob(inputs=val).mean()
#         loss.backward()
#         optimizer.step()
#     print(f"Avg. log prob = {-loss.item():.3f}")
#     scheduler.step()
    
FLOW_WEIGHTS = "../model_weights/contrastive_maf_flow.pt"
# print("Saving Flow")
# torch.save(flow_obj.state_dict(), FLOW_WEIGHTS)
flow_obj.load_state_dict(torch.load(FLOW_WEIGHTS, map_location=device))
print("FLOW LOADED")

anomaly_dataset = np.load('../data/bsm_datasets_-1.npz')

# For SM testing set
# pick individual events from the test set and get p-val, stop at 50K to be fast

sm_p_vals = []
for idx, val in enumerate(test_data_loader, 1):
    if val.shape[0] != test_batch_size:
        print(val.shape[0], test_batch_size)
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(val)
        sm_p_vals.append(log_p)
        

a_4l = data['ato4l']
    
a_4l = torch.from_numpy(a_4l.reshape(-1, latent_size)).to(dtype=torch.float32, device=device)
a_4l_loader = DataLoader(a_4l, shuffle=True, batch_size=test_batch_size)


a_4l_p_vals = []
for idx, val in enumerate(a_4l_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(val)
        a_4l_p_vals.append(log_p)
        
sm_p_vals = torch.stack(sm_p_vals).flatten()
a_4l_p_vals = torch.stack(a_4l_p_vals).flatten()
print(f"A to 4L p-val median {torch.median(a_4l_p_vals):.3f}")
print(f"A to 4L p-val mean {torch.mean(a_4l_p_vals):.3f}")
print(f"SM p-val median {torch.median(sm_p_vals):.3f}")
print(f"SM p-val mean {torch.mean(sm_p_vals):.3f}")

h_tau_tau = data['hToTauTau']
h_tau_tau = torch.from_numpy(h_tau_tau.reshape(-1, latent_size)).to(dtype=torch.float32, device=device)
h_tau_tau_loader = DataLoader(h_tau_tau, shuffle=True, batch_size=test_batch_size)

h_tau_tau_p_vals = []
for idx, val in enumerate(h_tau_tau_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(val)
        h_tau_tau_p_vals.append(log_p)
        
h_tau_tau_p_vals = torch.stack(h_tau_tau_p_vals).flatten()
print(f"h to tau-tau p-val median {torch.median(h_tau_tau_p_vals):.3f}")
print(f"h to tau-tau p-val mean {torch.mean(h_tau_tau_p_vals):.3f}")

h_tau_nu = data['hChToTauNu']
    
h_tau_nu = torch.from_numpy(h_tau_nu.reshape(-1, latent_size)).to(device=device, dtype=torch.float32)
h_tau_nu_loader = DataLoader(h_tau_nu, shuffle=True, batch_size=test_batch_size)

h_tau_nu_p_vals = []
for idx, val in enumerate(h_tau_nu_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(val)
        h_tau_nu_p_vals.append(log_p)
h_tau_nu_p_vals = torch.stack(h_tau_nu_p_vals).flatten()
print(f"h to tau-nu p-val median {torch.median(h_tau_nu_p_vals):.3f}")
print(f"h to tau-nu p-val mean {torch.mean(h_tau_nu_p_vals):.3f}")

leptoquark = data['leptoquark']
    
leptoquark = torch.from_numpy(leptoquark.reshape(-1, latent_size)).to(dtype=torch.float32, device=device)
leptoquark_loader = DataLoader(leptoquark, shuffle=True, batch_size=test_batch_size)

leptoquark_p_vals = []
for idx, val in enumerate(leptoquark_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(val)
        leptoquark_p_vals.append(log_p)
        
leptoquark_p_vals = torch.stack(leptoquark_p_vals).flatten()
print(f"LQ to b-tau p-val median {torch.median(leptoquark_p_vals):.3f}")
print(f"LQ to b-tau p-val mean {torch.mean(leptoquark_p_vals):.3f}")

plt.hist(sm_p_vals.cpu(), range=(-25, 6), bins=50, label='SM',
         density=True, alpha=0.8)
plt.hist(leptoquark_p_vals.cpu(), range=(-25, 6), bins=50, label='$LQ \\rightarrow b\\tau$',
         density=True, alpha=0.8, histtype='step', linewidth=2)
plt.hist(h_tau_tau_p_vals.cpu(), range=(-25, 6), bins=50, label='$h \\rightarrow \\tau\\tau$',
         density=True, alpha=0.8, histtype='step', linewidth=2)
plt.hist(h_tau_nu_p_vals.cpu(), range=(-25, 6), bins=50, label='$h^{\pm} \\rightarrow \\tau\\nu$',
         density=True, alpha=0.8, histtype='step', linewidth=2)
plt.hist(a_4l_p_vals.cpu(), range=(-25, 6), bins=50, label='$A \\rightarrow 4l$',
         density=True, alpha=0.8, histtype='step', linewidth=2)
plt.title("SM and BSM event p-vals")
plt.annotate("", xy=(-5.1, 0.5), xytext=(-7.5, 0.5),
             arrowprops=dict(arrowstyle="<-"))
plt.text(-7.5, 0.52, "Detection")
plt.xlabel("log prob.")
plt.legend()

plots_subfolder = os.path.join(os.path.dirname(__file__), '..', 'plots')
plots_subfolder = os.path.join(os.getcwd(), plots_subfolder, 'deep_replica_nov17')
os.makedirs(plots_subfolder, exist_ok=True)
file_path = os.path.join(plots_subfolder, 'contrastive_log_prob.png')
plt.savefig(file_path)
plt.close('all')
print(f"sm and bsm log prob saved")


from itertools import cycle
import numpy as np

colors = cycle(['red', 'green', 'blue', 'black'])
linestyles = cycle(['--', '-.', ':', '-'])

thresholds = np.linspace(-1000, 0, 500)

p_vals = {
    'h_tau_tau': h_tau_tau_p_vals.cpu().numpy(),
    'h_tau_nu': h_tau_nu_p_vals.cpu().numpy(),
    'A_4l': a_4l_p_vals.cpu().numpy(),
    'leptoquark': leptoquark_p_vals.cpu().numpy()
}

efficiencies = dict.fromkeys(p_vals)
false_alarms = dict.fromkeys(p_vals)


for k, p_val in p_vals.items():
    eff = []
    false = []
    for threshold in thresholds:
        _eff = np.sum(p_val < threshold)
        _false = np.sum(sm_p_vals.cpu().numpy() < threshold)
        eff.append(_eff)
        false.append(_false)
    efficiencies[k] = np.array(eff)/len(p_val)
    false_alarms[k] = np.array(false)/len(sm_p_vals)
    
plt.figure()

for k in efficiencies:
    plt.loglog(false_alarms[k], efficiencies[k], label=k,
               color=next(colors), linestyle=next(linestyles), alpha=0.6)

plt.legend()
plt.ylim((1e-4, 1))
plt.ylabel("Efficiency")
plt.xlabel("False Alarm")
plt.axvline(x=1e-5, c='r')
plt.grid()

file_path = os.path.join(plots_subfolder, 'contrastive_roc.png')
plt.savefig(file_path)
plt.close('all')
print(f"roc saved ")

