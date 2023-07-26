print("Importing from 'graphing_module.py'")
import os 
import corner 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
            if anomaly is not None: 
                if label != 4.0: alpha, color = 0.025, anomaly_colors[label]
                else: 
                    print('key', label, 'shape', indices.shape)
                    alpha, color = 0.0125, anomaly_colors[label]
            else: alpha, color = 0.050, default_colors[label]
            plt.scatter(components[indices, 0], components[indices, 1], label=name, alpha=0.05, c=color, s=0.7)
        
        leg = plt.legend(markerscale = 3.0, loc='upper right')
        for lh in leg.legendHandles: lh.set_alpha(1)

    else: plt.scatter(components[:, 0], components[:, 1], alpha=0.5, s=0.7)
        
    if specs is not None: plt.title(f'2D PCA E: {specs[0]} BS: {specs[1]} LR: {specs[2]}')
    elif anomaly is not None: plt.title(f'2D PCA Plot with {anomaly} anomaly')
    else: plt.title('2D PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Saves graph to directory + reports success
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.savefig(save_file)
    plt.close('all')
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
            if anomaly is not None: 
                if label != 4.0: alpha, color = 0.025, anomaly_colors[label]
                else: alpha, color = 0.0125, anomaly_colors[label]
            else: alpha, color = 0.025, default_colors[label]
            ax.scatter(components[indices, 0], components[indices, 1], components[indices, 2], 
                       label=name, alpha=alpha, c=color, s=0.7)
    
        # Creates labels. Sets label marker to greater opacity/size than graph
        leg = ax.legend(markerscale = 3.0, loc='upper right')
        for lh in leg.legendHandles: lh.set_alpha(1) 
    
    # If no labels, plots everything same color with lower opacity/size
    else: ax.scatter(components[:, 0], components[:, 1], components[:, 2], alpha=0.5, s=0.3)
    
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
    plt.close('all')
    print(f"3D PCA plot saved at '{save_file}'")
    
    
def plot_pca_proj(representations, filename, labels, specs = None, anomaly = None):
    '''
    Creates histogram components of corner plots on four primary PCA Components. 
    Color code histograms based on labels
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
    plt.close('all')
    print(f"PCA projection plot saved at '{save_file}'")
      
        
def plot_corner_plots(sample_representations, filename, sample_labels, plot_pca = False, anomaly = None):
    '''
    Creates full corner plots of representations seperated by labels. Saves resulting plot
    in filename location. 
    '''
    def normalize_weights(curr_rep_count, max_rep_count):
        ''' Small fxn to normalize hist weights respective to other counts'''
        return np.ones(curr_rep_count) * (max_rep_count / curr_rep_count)
    
    # Creates two lists by seperating all data by their labels
    name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:anomaly}
    representations, labels = [], []
    unique_labels = np.unique(sample_labels)
    
    if plot_pca == True: 
        print("Plotting PCA Comps")
        pca = PCA(n_components=3)
        sample_representations = pca.fit_transform(sample_representations)
    
    for label in unique_labels: 
        indices = np.where(sample_labels == label)[0]
        labels.append(name_mappings[label])
        representations.append(sample_representations[indices])
    
    CORNER_KWARGS = dict(smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=True,
        max_n_ticks=3)

    max_rep_count = max([len(s) for s in representations])
    num_labels, num_dims = len(representations), representations[0].shape[1]
    colors = ['deeppink', 'lightseagreen', 'red', 'green', 'blueviolet'][::-1]

    plot_range = []
    for dim in range(num_dims): # Finds min and max range in each dim for histogram plots
        plot_range.append([1.5 * min([min(representations[i].T[dim]) for i in range(num_labels)]),
                           1.5 * max([max(representations[i].T[dim]) for i in range(num_labels)]),])
    
    CORNER_KWARGS.update(range=plot_range) # Plots first rep to define baseline plot
    fig = corner.corner(representations[0], color=colors[0], **CORNER_KWARGS,
                        weights=normalize_weights(len(representations[0]), max_rep_count))
    
    for idx in range(1, num_labels): # Adds each subsequent rep to same plot and normalize counts
        fig = corner.corner(representations[idx], fig = fig, color=colors[idx], **CORNER_KWARGS,
                            weights=normalize_weights(len(representations[idx]), max_rep_count))
        
    plt.legend(fontsize=20, frameon=False, bbox_to_anchor=(1, num_dims), loc="upper right", 
              handles=[mlines.Line2D([], [], color=colors[i], label=labels[i]) for i in range(num_labels)])

    # Saves file, clear plots, and return success
    os.makedirs('Plots', exist_ok=True)
    save_file = os.path.join('Plots', filename)
    plt.savefig(save_file)
    plt.close('all')
    print(f"Corner Plots saved at '{save_file}'")
    
    
    
    