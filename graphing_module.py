print("Importing from 'graphing_module.py'")
import os 
import corner 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_curve, auc

def plot_2D_pca(representations, folder, filename, labels = None, anomaly=None, specs=None): 
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
                else: alpha, color = 0.0125, anomaly_colors[label]
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
    subfolder = os.path.join(os.getcwd(), 'Plots', folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path)
    plt.close('all')
    print(f"2D PCA plot saved at '{filename}'")

    
    
def plot_3D_pca(representations, folder, filename, labels = None, anomaly = None, specs = None):
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
    subfolder = os.path.join(os.getcwd(), 'Plots', folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path)
    plt.close('all')
    print(f"3D PCA plot saved at '{filename}'")
     
    
def plot_corner_plots(sample_representations, folder, filename, sample_labels, plot_pca = False, anomaly = None):
    '''
    Creates full corner plots of representations seperated by labels. Saves resulting plot
    in filename location. 
    '''
    print("Plotting Corner Plots!")
    def normalize_weights(curr_rep_count, max_rep_count):
        '''Normalize hist heights/weights respective to other bin counts'''
        return np.ones(curr_rep_count) * (max_rep_count / curr_rep_count)
    
    name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:anomaly}
    
    if plot_pca: # Updates rep to be 3 principle components instead of latent space
        pca = PCA(n_components=3) 
        sample_representations = pca.fit_transform(sample_representations)
    
    # Partitions data by event type
    representations, labels = [], []
    unique_labels = np.unique(sample_labels)
    for label in unique_labels[::-1]: 
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
    if anomaly == None: colors = ['deeppink', 'lightseagreen', 'red', 'green', 'blueviolet'][::-1]
    else: colors = ['deeppink', 'lightseagreen', 'gold', 'green', 'blueviolet'][::-1]

    plot_range = []
    for dim in range(num_dims): # Finds min and max range in each dim for histogram plots
            plot_range.append([-5,5,])
#         plot_range.append([1.5 * min([min(representations[i].T[dim]) for i in range(num_labels)]),
#                            1.5 * max([max(representations[i].T[dim]) for i in range(num_labels)]),])
    CORNER_KWARGS.update(range=plot_range) # Plots first rep to define baseline plot
    fig = corner.corner(representations[0], color=colors[0], **CORNER_KWARGS,
                        weights=normalize_weights(len(representations[0]), max_rep_count))
    
    for idx in range(1, num_labels): # Adds each subsequent rep to same plot and normalize counts
        fig = corner.corner(representations[idx], fig = fig, color=colors[idx], **CORNER_KWARGS,
                            weights=normalize_weights(len(representations[idx]), max_rep_count))
        
    plt.legend(fontsize=20, frameon=False, bbox_to_anchor=(1, num_dims), loc="upper right", 
              handles=[mlines.Line2D([], [], color=colors[i], label=labels[i]) for i in range(num_labels)])

    # Saves file, clear plots, and return success
    subfolder = os.path.join(os.getcwd(), 'Plots', folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path)
    plt.close('all')
    print(f"Corner Plots saved at '{filename}'")

    
def plot_ROC(predicted_backgrounds, predicted_anomalies, folder, filename, testing_anomaly_name): 
    '''
    Plots ROC Curves for classification analysis. 
    predicted_anomalies (dict): keys are anomaly names, values are corresponding predictions/loss val
    predicted_backgrounds: functionality depends on type
        - if dict: Compares different classifiers
        - if list: Comparing same classifier's performance on different anomalies
    '''
    print("Plotting ROC Plots!")   
    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']
    for ix, anomaly_name in enumerate(predicted_anomalies): 
        predicted_anomaly = predicted_anomalies[anomaly_name]
        
        # Computes predicted background based on type of analysis (see docstring for more details)
        if isinstance(predicted_backgrounds, dict): predicted_background = predicted_backgrounds[anomaly_name]
        else: predicted_background = predicted_backgrounds
            
        # Defines true and predicted classes. Assumes background true class is 0, anomaly true class is 1
        true_class = np.concatenate((np.zeros(predicted_background.shape[0]), np.ones(predicted_anomaly.shape[0])))
        predicted_class = np.concatenate((predicted_background, predicted_anomaly)) 
        
        # Calculates ROC properties and plots on same plt
        false_pos_rate, true_pos_rate, threshold = roc_curve(true_class, predicted_class)
        area_under_curve = auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label=f'{anomaly_name} with AUC: {area_under_curve*100:.0f}%', 
                 linewidth=2, color=colors[ix]) 
    
    # Defines plot properties to highlight plot properties relevant to FPGA constraints
    plt.xlim(10**(-6),1) 
    plt.ylim(10**(-6),1)
    plt.xscale('log') 
    plt.yscale('log') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    if isinstance(predicted_backgrounds, dict): plt.title(f'ROC Performance per Anomaly Used in Training')  
    else: plt.title(f'ROC Performance on BSM Data Using {testing_anomaly_name} for Training')
    
    # Creates x=y line to compare model against random classification performance
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=2)
    plt.legend(loc='lower right', frameon=False)
    
    # Saves plot and reports success 
    subfolder = os.path.join(os.getcwd(), 'Plots', folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path) 
    plt.close('all')
    print(f"ROC Plot saved at '{filename}'")
    
    
    
def plot_pca_proj(representations, folder, filename, labels, specs = None, anomaly = None):
    '''
    Creates histogram of four primary PCA Components. Color code histograms based on labels
    '''
    print("Plotting PCA Projection Plots!")
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
    subfolder = os.path.join(os.getcwd(), 'Plots', folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path)
    plt.close('all')
    print(f"PCA projection plot saved at '{filename}'")
    
    