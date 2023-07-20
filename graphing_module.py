import os 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

def plot_2D_pca(representations, filename, labels = None, anomaly=None, specs=None): 
    '''
    Plots 2D PCA of representations and saves file at filename location 
    Labels: Provided in supervised case -> PCA color coded by labels 
    Anomaly: Str representing name of anomaly if represented in data
    Specs: [epochs, batch size, learning rate]. Include info in title for hparam training clarity 
    '''
    print("Plotting 2D PCA!")
    pca = PCA(n_components=2)
    components = pca.fit_transform(representations)

    if labels is not None: 
        unique_labels = np.unique(labels)
        name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:anomaly}
        anomaly_colors = {0.0:'#00C142', 1.0:'#44CBB7', 2.0:'#4457CB', 3.0:'#8B15E4', 4.0:'#C01E1E'}
        default_colors = {0.0:'#1f77b4', 1.0:'#ff7f0e', 2.0:'#2ca02c', 3.0:'#d62728', 4.0:'#9467bd'}
        
        # Plots representation per label. If anomaly -> uses special color pallet + opacity
        for label in unique_labels: 
            name = name_mappings[label]
            indices = np.where(labels == label)[0]
            if anomaly is not None: alpha, color = 0.025, anomaly_colors[label]
            else: alpha, color = 0.050, default_colors[label]
            plt.scatter(components[indices, 0], components[indices, 1], label=name, alpha=0.05, c=color, s=0.7)
        
        leg = plt.legend(markerscale = 3.0)
        for lh in leg.legendHandles: lh.set_alpha(1)

    else: plt.scatter(components[:, 0], components[:, 1], alpha=0.025, s=0.7)
        
    if specs is not None: plt.title(f'2D PCA E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    else: plt.title('2D PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Saves graph to directory + reports success
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.savefig(save_file)
    print(f"2D PCA plot saved at '{save_file}'")

def plot_3D_pca(representations, filename, labels = None, anomaly = None, specs = None):
    '''
    Plots 3D PCA of representations and saves file at filename location 
    Labels: Provided in supervised case -> PCA color coded by labels 
    Anomaly: Str representing name of anomaly if represented in data
    Specs: [epochs, batch size, learning rate]. Include info in title for hparam training clarity 
    '''
    # Perform PCA to extract the 3 prin. components
    print("Plotting 3D PCA!")
    pca = PCA(n_components=3)
    components = pca.fit_transform(representations)
    
    # Creates three dimensional plots 
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    if labels is not None: 
        unique_labels = np.unique(labels)
        name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:anomaly}
        anomaly_colors = {0.0:'#00C142', 1.0:'#44CBB7', 2.0:'#4457CB', 3.0:'#8B15E4', 4.0:'#C01E1E'}
        default_colors = {0.0:'#1f77b4', 1.0:'#ff7f0e', 2.0:'#2ca02c', 3.0:'#d62728', 4.0:'#9467bd'}

        # Plots representation per label. If anomaly -> uses special color pallet + opacity
        for label in unique_labels: 
            name = name_mappings[label]
            indices = np.where(labels == label)[0]
            if anomaly is not None: alpha, color = 0.025, anomaly_colors[label]
            else: alpha, color = 0.050, default_colors[label]
            ax.scatter(components[indices, 0], components[indices, 1], components[indices, 2], 
                       label=name, alpha=alpha, c=color, s=0.7)
    
        # Creates labels. Sets label marker to greater opacity/size than graph
        leg = ax.legend(markerscale = 3.0)
        for lh in leg.legendHandles: lh.set_alpha(1) 
    
    # If no labels, plots everything same color with lower opacity/size
    else: ax.scatter(components[:, 0], components[:, 1], components[:, 2], alpha=0.025, s=0.3)
    
    # Labels axis and creates title 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    if anomaly is not None: ax.set_title(f'3D PCA Plot with {anomaly} anomaly')
    elif specs is not None: ax.set_title(f'3D PCA E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    else: ax.set_title(f'3D PCA Plot')
    
    # Saves graph to directory + reports success
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.savefig(save_file)
    print(f"3D PCA plot saved at '{save_file}'")
    
    
def plot_pca_proj(representations, filename, labels, specs = None, anomaly = None):
    '''
    Creates modified corner plots of four PCA Components. Color code based on labels
    '''
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
            if anomaly is not None: color = anomaly_colors[label]
            else: color = default_colors[label]
            name = name_mapping[label]
            ax.hist(components[index_mapping[label], ix], bins=100, alpha=0.75, label=name, color=color)
        ax.legend()
        ax.set_ylabel("Frequency")
        ax.set_xlabel(f"Principal Component {ix+1}")
    
    # Saves graph to directory + reports success    
    if specs is not None: fig.suptitle(f'PCA Projections E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    elif anomaly is not None: fig.suptitle(f"PCA Projections with '{anomaly}' anomaly")
    else: fig.suptitle('PCA Projections')
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.clf()
    plt.rcdefaults()
    print(f"PCA projection plot saved at '{save_file}'")
    
    
def make_feature_histogram_plot(real, reco, particle, feature, output_dir):
    '''
    For VAE + Reconstruction Accuracy evaluation. Makes histogram of reconstruction vs
    real data for specific particle and feature 
    '''
    # Filters real/reco arrays to be feature specific
    feature_mapping = {"pT": 0, "eta":1, "phi":2}
    feature_ix = feature_mapping[feature]
    real = real[:, feature_ix]
    reco = reco[:, feature_ix]

    # zero features -> no data -> use masking
    zeroes = np.where(real == 0)[0]
    real = np.delete(real, zeroes)
    reco = np.delete(reco, zeroes)

    if feature == "eta": # eta trick to enforce physical cosntraints
        reco = 3.0 * np.tanh(reco) if particle == "Electrons" else reco
        reco = 2.1 * np.tanh(reco) if particle == "Muons" else reco
        reco = 4.0 * np.tanh(reco) if particle == "Jets" else reco

    if feature == "phi": #phi trick to enforce physical constraints
        reco = math.pi * np.tanh(reco)

    fig, ax = plt.subplots(2)

    # Plot two histograms for real and reco probability density.
    ax[0].hist(real, bins=100, density=True, histtype="step", linewidth=1.5, label=f'{particle} True')
    ax[0].hist(reco, bins=100, density=True, histtype="step", linewidth=1.5, label=f'{particle} Predicted')
    ax[0].set_ylabel('Prob. Density (a.u.)', fontsize=12)
    ax[0].set_xlabel(feature, fontsize=12)
    ax[0].legend(fontsize=12, frameon=False)

    if feature == "pT": # axis scaling for pT ftr
        ax[0].set_yscale('log', nonpositive='clip')

    if feature != "eta" and particle != "MET":
        pull = (real-reco)/real # Finds pull -> Ask Katya abt
        rmin = np.min(pull) if np.min(pull)>-1000 else -1000
        rmax = np.max(pull) if np.max(pull)< 1000 else  1000
        n, bins, _ = ax[1].hist(pull, 100, range=(int(rmin), int(rmax)), density=False, 
                                histtype='step', fill=False, linewidth=1.5, label=f'{particle} Pull')

    # Finds mean and std of pull to plot
    mids = 0.5 * (bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    var = np.average((mids-mean)**2, weights=n)
    std = np.sqrt(var)
    ax[1].legend(fontsize=12, frameon=False, title=f'mean={mean:0.2} RMS={std:0.2}')
    ax[1].set_yscale('log', nonpositive='clip')
    ax[1].set_xlabel(feature, fontsize = 12)

    fig.tight_layout()
    # fig.savefig(os.path.join(output_dir, f'{particle}_{feature}.pdf'))
    
def make_batch_feature_plots(real, reco, output_dir):
    '''
    For VAE + Reconstruction Accuracy evaluation. Makes histogram of reconstruction vs
    real for each particle (MET, egamma, muon, jet) and each feature (pT, n, phi) 
    '''
    for feature in ["pT", "eta", "phi"]:
        make_feature_histogram_plot(real[:, 0:3], reco[:, 0:3], "MET", feature, output_dir)
        make_feature_histogram_plot(real[:, 3:6], reco[:, 3:6], "Electrons", feature, output_dir)
        make_feature_histogram_plot(real[:, 15:18], reco[:, 15:18], "Muons", feature, output_dir)
        make_feature_histogram_plot(real[:, 27:30], reco[:, 27:30], "Jets", feature, output_dir)
    
def plot_rocs(standard_loss, anomalous_loss, bsm_name, i, axs):
    '''
    For VAE + Reconstruction Accuracy evaluation. Makes ROC Plots of VAE on standard v anomalous data
    '''
    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

    def get_roc_metrics(anomalous, standard):
        real_class = np.concatenate((np.ones(anomalous.shape[0]), np.zeros(standard.shape[0])))
        reco_class = np.concatenate((anomalous, standard))
        fpr, tpr, threshold = roc_curve(real_class, reco_class)
        auc_data = auc(fpr, tpr)
        return fpr, tpr, auc_data

    fpr, tpr, auc_data = get_roc_metrics(anomalous_loss, standard_loss)
    roc_model = axs[i].plot(fpr, tpr, label=f'model AUC = {auc_data*100:.0f}%', linewidth=2, color=colors[i])

    axs[i].set_xlim(10**(-6),1)
    axs[i].set_ylim(10**(-6),1.2)
    axs[i].semilogx()
    axs[i].semilogy()
    axs[i].set_ylabel('True Positive Rate', )
    axs[i].set_xlabel('False Positive Rate', )
    axs[i].plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=2)
    axs[i].legend(loc='lower right', frameon=False, title=f'ROC {bsm_name}', )
