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

X_test = features_dataset['x_test']
X_train = features_dataset['x_train']
X_val = features_dataset['x_val']
print("hello")
X_test = torch.from_numpy(X_test.reshape(-1, 19, 3)).to(dtype=torch.float32, device=device).transpose(1, 2)
print("my name is Mia")
X_train = torch.from_numpy(X_train.reshape(-1, 19, 3)).to(dtype=torch.float32, device=device).transpose(1, 2)
print("meow")
X_val = torch.from_numpy(X_val.reshape(-1, 19, 3)).to(dtype=torch.float32, device=device).transpose(1, 2)

print(f"Num dimensions {X_train.shape[-1]}; Num channels {X_train.shape[-2]}")

class HEPDataSet(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset.clone()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]

train_set = HEPDataSet(X_train)
val_set = HEPDataSet(X_val)
test_set = HEPDataSet(X_test)




batch_size = 4000
test_batch_size = 10000

train_data_loader = DataLoader(
    train_set, batch_size=batch_size,
    shuffle=True,
)
val_data_loader = DataLoader(
    val_set, batch_size=batch_size,
    shuffle=True,
)

test_data_loader = DataLoader(
    test_set, batch_size=test_batch_size,
    shuffle=True,
)

print(f"Num batches for training = {len(train_data_loader)}")

for idx, val in enumerate(train_data_loader):
    break

    
print("Val shape", val.shape)

# Simplified implemented from https://github.com/violatingcp/codec/blob/main/losses.py
class VICRegLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)

        x_mu = x.mean(dim=0)
        x_std = x.std(dim=0) + 1e-2
        y_mu = y.mean(dim=0)
        y_std = y.std(dim=0) + 1e-2

        x = (x - x_mu)/x_std
        y = (y - y_mu)/y_std

        N = x.size(0)
        D = x.size(-1)

        std_loss = torch.mean(F.relu(1 - x_std, inplace=False)) / 2
        std_loss += torch.mean(F.relu(1 - y_std, inplace=False)) / 2

        cov_x = (x.transpose(1, 2).contiguous() @ x) / (N - 1)
        cov_y = (y.transpose(1, 2).contiguous() @ y) / (N - 1)

        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        return repr_loss + cov_loss + std_loss

    def off_diagonal(self, x):
        num_batch, n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()
    
    
    
vicreg_loss = VICRegLoss()

from nflows.nn.nets import ResidualNet
from nflows import transforms, distributions, flows

class SimilarityEmbedding(nn.Module):
    """An embedding with resnets"""
    def __init__(self):
        super().__init__()
        self.resnet = ResidualNet(19, 1, 35)

    def forward(self, x):
        res_embedding = self.resnet(x)
        return res_embedding
    
similarity_embedding = SimilarityEmbedding().to(device)

optimizer = torch.optim.Adam(similarity_embedding.parameters(), lr=1e-4)

scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=5)
scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 20])

print("Similarity embedding shape", similarity_embedding(val).shape)

def train_one_epoch(epoch_index, tb_writer):
    running_sim_loss = 0.
    last_sim_loss = 0.

    for idx, val in enumerate(train_data_loader, 1):
        # only applicable to the final batch
        if val.shape[0] != batch_size:
            continue

        # embed entire batch with first value of the batch repeated
        first_val_repeated = val[0].repeat(batch_size, 1, 1)

        embedded_values_aug = similarity_embedding(first_val_repeated)
        embedded_values_orig = similarity_embedding(val)

        similar_embedding_loss = vicreg_loss(embedded_values_aug, embedded_values_orig)

        optimizer.zero_grad()
        similar_embedding_loss.backward()
        optimizer.step()
        # Gather data and report
        running_sim_loss += similar_embedding_loss.item()
        if idx % 500 == 0:
            last_sim_loss = running_sim_loss / 500
            print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_sim_loss))
            tb_x = epoch_index * len(train_data_loader) + idx
            tb_writer.add_scalar('SimLoss/train', last_sim_loss, tb_x)
            running_sim_loss = 0.
    return last_sim_loss


def val_one_epoch(epoch_index, tb_writer):
    running_sim_loss = 0.
    last_sim_loss = 0.

    for idx, val in enumerate(val_data_loader, 1):
        if val.shape[0] != batch_size:
            continue

        first_val_repeated = val[0].repeat(batch_size, 1, 1)

        embedded_values_aug = similarity_embedding(first_val_repeated)
        embedded_values_orig = similarity_embedding(val)

        similar_embedding_loss = vicreg_loss(embedded_values_aug, embedded_values_orig)

        running_sim_loss += similar_embedding_loss.item()
        if idx % 50 == 0:
            last_sim_loss = running_sim_loss / 50
            tb_x = epoch_index * len(val_data_loader) + idx + 1
            tb_writer.add_scalar('SimLoss/val', last_sim_loss, tb_x)
            tb_writer.flush()
            running_sim_loss = 0.
    tb_writer.flush()
    return last_sim_loss


epoch_number = 0
writer = SummaryWriter("hep_sim_together_round_2", comment="Similarity with LR=1e-3", flush_secs=5)

# EPOCHS = 10

# print("beginning training")
# for epoch in range(EPOCHS):
#     print('EPOCH {}:'.format(epoch_number + 1))
#     # Gradient tracking
#     similarity_embedding.train(True)
#     avg_train_loss = train_one_epoch(epoch_number, writer)
    
#     # no gradient tracking, for validation
#     similarity_embedding.train(False)
#     avg_val_loss = val_one_epoch(epoch_number, writer)
    
#     print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

#     epoch_number += 1
#     scheduler.step()
    
PATH = "../model_weights/deep_similarity_model.pt"
#  torch.save(similarity_embedding.state_dict(), PATH)
similarity_embedding.load_state_dict(torch.load(PATH, map_location=device))   

    
sim_vals = []

for idx, val in enumerate(test_data_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        sim_val = similarity_embedding(val)
        sim_vals.append(sim_val)
        
sim_vals = torch.stack(sim_vals).reshape(len(sim_vals)*test_batch_size, 3)

# helper function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Num trainable parameters in embedding net: {count_parameters(similarity_embedding)}")

# Freeze weights for embedding net
for name, param in similarity_embedding.named_parameters():
    param.requires_grad = False
    
    
    
### NORMLAIZING FLOW 

num_layers = 4
base_dist = distributions.StandardNormal(shape=[3])

transform_layers = []
for _ in range(num_layers):
    transform_layers.append(transforms.permutations.ReversePermutation(features=3))
    transform_layers.append(
        transforms.autoregressive.MaskedAffineAutoregressiveTransform(
            features=3,
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

#         sim_val = similarity_embedding(val).reshape(4000, 3)
#         optimizer.zero_grad()
#         loss = -flow_obj.log_prob(inputs=sim_val).mean()
#         loss.backward()
#         optimizer.step()
#     print(f"Avg. log prob = {-loss.item():.3f}")
#     scheduler.step()
    
FLOW_WEIGHTS = "../model_weights/deep_maf_flow.pt"
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
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(similarity_embedding(val).reshape(test_batch_size, 3))
        sm_p_vals.append(log_p)
        

a_4l = anomaly_dataset['ato4l']
    
a_4l = HEPDataSet(torch.from_numpy(a_4l.reshape(-1, 19, 3)).to(dtype=torch.float32, device=device).transpose(1, 2))
a_4l_loader = DataLoader(a_4l, shuffle=True, batch_size=test_batch_size)


a_4l_p_vals = []
for idx, val in enumerate(a_4l_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(similarity_embedding(val).reshape(test_batch_size, 3))
        a_4l_p_vals.append(log_p)
        
sm_p_vals = torch.stack(sm_p_vals).flatten()
a_4l_p_vals = torch.stack(a_4l_p_vals).flatten()
print(f"A to 4L p-val median {torch.median(a_4l_p_vals):.3f}")
print(f"A to 4L p-val mean {torch.mean(a_4l_p_vals):.3f}")
print(f"SM p-val median {torch.median(sm_p_vals):.3f}")
print(f"SM p-val mean {torch.mean(sm_p_vals):.3f}")

h_tau_tau = anomaly_dataset['hToTauTau']
h_tau_tau = HEPDataSet(torch.from_numpy(h_tau_tau.reshape(-1, 19, 3)).to(dtype=torch.float32, device=device).transpose(1, 2))
h_tau_tau_loader = DataLoader(h_tau_tau, shuffle=True, batch_size=test_batch_size)

h_tau_tau_p_vals = []
for idx, val in enumerate(h_tau_tau_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(similarity_embedding(val).reshape(test_batch_size, 3))
        h_tau_tau_p_vals.append(log_p)
        
h_tau_tau_p_vals = torch.stack(h_tau_tau_p_vals).flatten()
print(f"h to tau-tau p-val median {torch.median(h_tau_tau_p_vals):.3f}")
print(f"h to tau-tau p-val mean {torch.mean(h_tau_tau_p_vals):.3f}")

h_tau_nu = anomaly_dataset['hChToTauNu']
    
h_tau_nu = HEPDataSet(torch.from_numpy(h_tau_nu.reshape(-1, 19, 3)).to(device=device, dtype=torch.float32).transpose(1, 2))
h_tau_nu_loader = DataLoader(h_tau_nu, shuffle=True, batch_size=test_batch_size)

h_tau_nu_p_vals = []
for idx, val in enumerate(h_tau_nu_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(similarity_embedding(val).reshape(test_batch_size, 3))
        h_tau_nu_p_vals.append(log_p)
h_tau_nu_p_vals = torch.stack(h_tau_nu_p_vals).flatten()
print(f"h to tau-nu p-val median {torch.median(h_tau_nu_p_vals):.3f}")
print(f"h to tau-nu p-val mean {torch.mean(h_tau_nu_p_vals):.3f}")

leptoquark = anomaly_dataset['leptoquark']
    
leptoquark = HEPDataSet(torch.from_numpy(leptoquark.reshape(-1, 19, 3)).to(dtype=torch.float32, device=device).transpose(1, 2))
leptoquark_loader = DataLoader(leptoquark, shuffle=True, batch_size=test_batch_size)

leptoquark_p_vals = []
for idx, val in enumerate(leptoquark_loader, 1):
    if val.shape[0] != test_batch_size:
        continue
    with torch.no_grad():
        log_p = flow_obj.log_prob(similarity_embedding(val).reshape(test_batch_size, 3))
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
plt.axvline(-5, label='Ex. Threshold', color='red')
plt.annotate("", xy=(-5.1, 0.5), xytext=(-7.5, 0.5),
             arrowprops=dict(arrowstyle="<-"))
plt.text(-7.5, 0.52, "Detection")
plt.xlabel("log prob.")
plt.legend()

plots_subfolder = os.path.join(os.path.dirname(__file__), '..', 'plots')
plots_subfolder = os.path.join(os.getcwd(), plots_subfolder, 'deep_replica_nov17')
os.makedirs(plots_subfolder, exist_ok=True)
file_path = os.path.join(plots_subfolder, 'log_prob.png')
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

file_path = os.path.join(plots_subfolder, 'roc.png')
plt.savefig(file_path)
plt.close('all')
print(f"roc saved ")

