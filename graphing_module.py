import os 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

def plot_2D_pca(representations, labels, filename, anomaly=None, specs=None): 
    # Perform PCA to extract the two prin. components
    print("inside 2d creating  plots")
    pca = PCA(n_components=2)
    components = pca.fit_transform(representations)

    unique_labels = np.unique(labels)
    name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:anomaly}
    anomaly_colors = {0.0:'#00C142', 1.0:'#44CBB7', 2.0:'#4457CB', 3.0:'#8B15E4', 4.0:'#C01E1E'}
    default_colors = {0.0:'#1f77b4', 1.0:'#ff7f0e', 2.0:'#2ca02c', 3.0:'#d62728', 4.0:'#9467bd'}
    
    # Plots representation per label. If anomaly -> uses special color pallet + opacity
    for label in unique_labels: 
        name = name_mappings[label]
        indices = np.where(labels == label)[0]
        if anomaly: alpha, color = 0.025, anomaly_colors[label]
        else: alpha, color = 0.050, default_colors[label]
        plt.scatter(components[indices, 0], components[indices, 1], label=name, alpha=0.05, c=color, s=0.7)
        
    if specs: plt.title(f'2D PCA E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    else: plt.title('2D PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    leg = plt.legend(markerscale = 3.0)
    for lh in leg.legendHandles: lh.set_alpha(1)
    
    # Saves graph to directory + reports success
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.savefig(save_file)
    plt.clf()
    plt.rcdefaults()
    print(f"2D PCA plot saved at '{save_file}'")

def plot_3D_pca(representations, labels, filename, anomaly = None, specs = None):
    # Perform PCA to extract the 3 prin. components
    pca = PCA(n_components=3)
    components = pca.fit_transform(representations)

    unique_labels = np.unique(labels)
    name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:anomaly}
    anomaly_colors = {0.0:'#00C142', 1.0:'#44CBB7', 2.0:'#4457CB', 3.0:'#8B15E4', 4.0:'#C01E1E'}
    default_colors = {0.0:'#1f77b4', 1.0:'#ff7f0e', 2.0:'#2ca02c', 3.0:'#d62728', 4.0:'#9467bd'}
   
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    # Plots representation per label. If anomaly -> uses special color pallet + opacity
    for label in unique_labels: 
        name = name_mappings[label]
        indices = np.where(labels == label)[0]
        if anomaly: alpha, color = 0.025, anomaly_colors[label]
        else: alpha, color = 0.050, default_colors[label]
        ax.scatter(components[indices, 0], components[indices, 1], components[indices, 2], 
                   label=name, alpha=alpha, c=color, s=0.7)
    
    # Creates labels. Sets label marker to greater opacity/size than graph
    leg = ax.legend(markerscale = 3.0)
    for lh in leg.legendHandles: lh.set_alpha(1) 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    if anomaly: ax.set_title(f'3D PCA Plot with {anomaly} anomaly')
    elif specs: ax.set_title(f'3D PCA E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    else: ax.set_title(f'3D PCA Plot')
    
    # Saves graph to directory + reports success
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.savefig(save_file)
    plt.clf()
    plt.rcdefaults()
    print(f"3D PCA plot saved at '{save_file}'")
    
def plot_pca_proj(representations, labels, filename, specs = None, anomaly = None):
    # Perform PCA to extract the 3 prin. components
    pca = PCA(n_components=4)
    components = pca.fit_transform(representations)
    fig, axs = plt.subplots(2, 2, figsize=(12,18))
    
    unique_labels = np.unique(labels)
    index_mapping = {label: np.where(labels == label)[0] for label in unique_labels}
    
    name_mapping = {0.0: "W-Boson", 1.0: "QCD", 2.0: "Z_2", 3.0: "tt", 4.0: anomaly}
    default_colors = {0.0:'#1f77b4', 1.0:'#ff7f0e', 2.0:'#2ca02c', 3.0:'#d62728', 4.0:'#9467bd'}
    anomaly_colors = {0.0:'#00C142', 1.0:'#44CBB7', 2.0:'#4457CB', 3.0:'#8B15E4', 4.0:'#C01E1E'}

    # Plots histogram per label on same subplot
    for ix, ax in enumerate(axs.flatten()): 
        for label in unique_labels: 
            if anomaly: color = anomaly_colors[label]
            else: color = default_colors[label]
            name = name_mapping[label]
            ax.hist(components[index_mapping[label], ix], bins=100, alpha=0.75, label=name, color=color)
        ax.legend()
        ax.set_ylabel("Frequency")
        ax.set_xlabel(f"Principal Component {ix+1}")
    
    # Saves graph to directory + reports success    
    if specs: fig.suptitle(f'PCA Projections E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    elif anomaly: fig.suptitle(f"PCA Projections with '{anomaly}' anomaly")
    else: fig.suptitle('PCA Projections')
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.clf()
    plt.rcdefaults()
    print(f"PCA projection plot saved at '{save_file}'")
    
    
