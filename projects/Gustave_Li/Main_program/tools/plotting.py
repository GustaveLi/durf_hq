from bokeh.plotting import figure, show
from bokeh.models.tools import *
from bokeh.models import ColumnDataSource, ColorBar, Title, Label, LabelSet, BoxAnnotation
from bokeh.layouts import row
from bokeh.io import output_notebook, export_png
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Plasma256, Set1
import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md

file_dir = '/xspace/hl4212/DURF_datasets/triad_molecule'
traj_path = f'{file_dir}/triad_dataset_aligned.nc'
top_path = f'{file_dir}/triad_forcefield_ground.prmtop'
traj=md.load(traj_path, top=top_path)

def RMSD_map(array, title='RMSD_map', ref_frame=0, save_dir=None):
    """
     Visualize the results from given 2D array, color each point with rmsd value

    Parameters
    ----------
    title : TYPE Str, optional
        DESCRIPTION. Specify the title of the plot. The default is 'RMSD_map'.
    array : TYPE Numpy array, shape=(num of instances, 2)
        DESCRIPTION. The array used for plotting (dimension reduced to 2)
    save_dir : TYPE file path (string), optional
        DESCRIPTION. If specified, the plot will be saved to the given directory. The default is None.
    ref_frame : TYPE int, optional
        DESCRIPTION. The reference frame for RMSD calculation. The default is 0.

    Returns
    -------
    None. Prints (and saves) the plot

    """
    # check for dimensionality
    if array.shape[-1]==2:        
        x = array[:, 0].reshape(len(array), )
        y = array[:, 1].reshape(len(array), )
        rmsd_array = md.rmsd(traj, traj, frame=ref_frame)
        source = ColumnDataSource(dict(x=x,y=y, rmsd_array=rmsd_array))
        mapper = linear_cmap(field_name='rmsd_array', palette=Plasma256 ,\
                             low=min(rmsd_array) ,high=max(rmsd_array))
        p = figure(title=title, 
                   x_axis_label='dim1', 
                   y_axis_label='dim2', 
                   tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), \
                          SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')]
                  )
        p.circle(x='x', y='y', line_color=mapper, fill_color=mapper, 
                 source=source, alpha=0.1, size=3)
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
        p.add_layout(color_bar, 'right')
        p.add_layout(Title(text="RMSD", align="center"), "right")
        output_notebook()
        show(p)
        if save_dir != None:
            export_png(p, filename=f'{save_dir}/{title}.png')
    else:
        print('Sorry, only dimension reduced to 2D accepted.')
        
def cluster_map(num_instances, dimred_method, clustering_method, index, 
                threshold=None, exclude_neg_1=False, save_dir=None):
    """
    Displays differnet clusters in different colors, highlight the position 
    of cluster centers (if provided)

    Parameters
    ----------
    num_instances : TYPE Integer (10000, 40000, 70000 or 100000)
        DESCRIPTION. Number of instances used for clustering
    dimred_method : TYPE String
        DESCRIPTION. Method used for dimensionality reduction
    clustering_method : TYPE String
        DESCRIPTION. Method used for clustering
    index : TYPE Integer
        DESCRIPTION. The hyperparameter index from the preclustering loop,
        can be obtained by inspecting the benchmark plot
    threshold : TYPE Float, optional
        DESCRIPTION. The RMSD threshold for high dimensional clustering.
        Only use this parameter when the clusters are generated from high dimensioanl clustering.
        The default is None.
    exclude_neg_1 : TYPE Bool, optional
        DESCRIPTION. If True, exclude instances with label of -1 when plotting.
        Only use this parameter when the clusters are generated from high dimensioanl clustering.
        The default is False.
    save_dir : TYPE File path (string), optional
        DESCRIPTION. If specified, the plot will be saved to the given 
        directory. The default is None.

    Returns
    -------
    None. Prints (and saves) the plot

    """
    
    # Load files: feature space data, label array, center array(optional)
    results_dir = '/xspace/hl4212/results'
    instance_path = f'{results_dir}/dimensionality_reduction/dimreduct_{dimred_method}.npy'
    instance_array = np.load(instance_path)[:num_instances, :]
    if threshold != None:
        label_path = f'{results_dir}/high_dim_clustering/{dimred_method}_{clustering_method}_{index}_{threshold}_Labels.npy'
        label_array = np.load(label_path)
    else:
        label_path = f'{results_dir}/clustering/{dimred_method}_{clustering_method}_{num_instances}_labels.npy'
        label_array = np.load(label_path)[index]   
    
    try:
        if threshold != None:
            center_idx_path = f'{results_dir}/high_dim_clustering/{dimred_method}_{clustering_method}_{index}_{threshold}_CenterIdx.npy'
            center_idx_array = np.load(center_idx_path)
            center_array = instance_array[center_idx_array]     
        else:
            center_path = f'{results_dir}/clustering/{dimred_method}_{clustering_method}_{num_instances}_ccenters.npy'
            center_array = np.load(center_path, allow_pickle=True)[index]
        
    except:
        pass
    
    if exclude_neg_1 == True:
        instance_array = instance_array[np.where(label_array >= 0)]
        label_array = label_array[np.where(label_array >= 0)]
        
    x = instance_array[:, 0].reshape(len(instance_array), )
    y = instance_array[:, 1].reshape(len(instance_array), )
    
    if -1 in label_array:
        total_cluster = len(np.unique(label_array))-1
    else:
        total_cluster = len(np.unique(label_array))
        
    # First plot the feature space data and color the points according to their labels
    source = ColumnDataSource(dict(x=x,y=y, label_array=label_array.astype(str)))
    label_idx = np.unique(label_array)
    mapper = factor_cmap('label_array', palette = Set1[len(label_idx)], factors=label_idx.astype(str))
    
    p = figure(title=f'{dimred_method}_{clustering_method}_{num_instances}', 
               x_axis_label='dim1', 
               y_axis_label='dim2', 
               tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), \
                      SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
               plot_height = 800,
               plot_width = 800
               )
    p.circle(x='x', y='y', line_color=mapper, fill_color=mapper,
             source=source, alpha=0.7,size=3, legend_field='label_array')
    
    # Add a label in the bottom middle of the plot, showing the total number of clusters
    label = Label(x=200, y=10, x_units='screen', y_units='screen',
                 text=f'Total number of clusters: {total_cluster}',
                 border_line_color='black', border_line_alpha=1.0,text_line_height=1.5,
                 background_fill_color='White', text_font_size='20px')
    p.add_layout(label)
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    
    # If cluster center provided, highlight the position of cluster centers, and label the index
    try:
        x_center = center_array[:, 0].reshape(len(center_array), )
        y_center = center_array[:, 1].reshape(len(center_array), )
        index_ = list(range(len(x_center)))
        ccsource = ColumnDataSource(data=dict(x=x_center, y=y_center, index_=index_))
        p.triangle('x', 'y', line_color='white', fill_color='black', size=10, source=ccsource, 
                   legend_label = 'Cluster center (cluster index)')
        labels = LabelSet(x='x', y='y', text='index_', x_offset=5, y_offset=5, render_mode='canvas', source=ccsource,
                         background_fill_color='white', background_fill_alpha = 1, text_color = 'black',
                         text_font_size='30px', text_font_style='bold')
        p.add_layout(labels)
        
    except:
        pass

    output_notebook()
    show(p)
    
    if save_dir != None:
        export_png(p, filename=f'{save_dir}/{dimred_method}_{clustering_method}_{index}.png')
        
def preclustering_scoring(x_axis, dimreduct_method, clustering_method, all_runs = True, tags=None, save_dir = None):
    """
    Plot the benchmark score (AIC, BIC, Inertia, Silhouette score) with respect 
    to differnet variables (cluster_num, instances_num). Results generated from
    different sample sizes will be inspected and compared in one graph.

    Parameters
    ----------
    x_axis : TYPE 1D Numpy array
        DESCRIPTION. The x-axis value (cluster_num or sequence)
    dimreduct_method : TYPE String
        DESCRIPTION. Method used for dimensionality reduction
    clustering_method : TYPE String
        DESCRIPTION. Method used for clustering
    all_runs: TYPE Bool, optional
        DESCRIPTION If set to True, Results generated from
        different sample sizes will be inspected and compared in one graph.
        If set to False, only results with file size 100,000 will be displayed.
        The default is True.
    tags : TYPE 1D Numpy array, optional
        DESCRIPTION The parameter tags, specific to HDBSCAN algorithm where 
        there are multiple parameters. The default is None.

    Returns
    -------
    None.
        Prints the following plots on screen: 
        For kmeans/kmedoids (a=2), two subplots, inertia-x & silhouette score-x.
        For gmm (a=3), three subplots, AIC-x, BIC-x & silhouette score-x.
        For hdbscan (a=1), one plot, silhouette score-x.

    """
    read_dir = '/xspace/hl4212/results/clustering_rough'
    if all_runs == True:
        dataset_10000 = np.load(f'{read_dir}/{dimreduct_method}_{clustering_method}_10000_benchmarks.npy')   
        dataset_40000 = np.load(f'{read_dir}/{dimreduct_method}_{clustering_method}_40000_benchmarks.npy')
        dataset_70000 = np.load(f'{read_dir}/{dimreduct_method}_{clustering_method}_70000_benchmarks.npy')
    dataset_100000 = np.load(f'{read_dir}/{dimreduct_method}_{clustering_method}_100000_benchmarks.npy')

    if len(dataset_100000[0]) == 2:
        p1 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    title=f'Inertia_{dimreduct_method}_{clustering_method}',
                    tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), 
                           SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
                    )
        p2 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    title=f'Silhouette score_{dimreduct_method}_{clustering_method}',
                    tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), 
                           SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
                    )
        
        if all_runs == True:
            p1.line(x=x_axis, y=dataset_10000[: , 0], legend_label = 'num of instances: 10000', 
                    line_width=2, line_color='blue')
            p1.line(x=x_axis, y=dataset_40000[: , 0], legend_label = 'num of instances: 40000', 
                    line_width=2, line_color='red')
            p1.line(x=x_axis, y=dataset_70000[: , 0], legend_label = 'num of instances: 70000', 
                    line_width=2, line_color='green')
        p1.line(x=x_axis, y=dataset_100000[: , 0], legend_label = 'num of instances: 100000', 
                line_width=2, line_color='grey')
        
        if all_runs == True:
            p2.line(x=x_axis, y=dataset_10000[: , 1], legend_label = 'num of instances: 10000', 
                    line_width=2, line_color='blue')
            p2.line(x=x_axis, y=dataset_40000[: , 1], legend_label = 'num of instances: 40000', 
                    line_width=2, line_color='red')
            p2.line(x=x_axis, y=dataset_70000[: , 1], legend_label = 'num of instances: 70000', 
                    line_width=2, line_color='green')
        p2.line(x=x_axis, y=dataset_100000[: , 1], legend_label = 'num of instances: 100000', 
                line_width=2, line_color='grey')
        
        output_notebook()
        show(row(p1,p2))
        if save_dir != None:
            export_png(p1, filename=f'{save_dir}/Inertia_{dimreduct_method}_{clustering_method}.png')
            export_png(p2, filename=f'{save_dir}/Silhouette_score_{dimreduct_method}_{clustering_method}.png')
        
    elif len(dataset_100000[0]) == 3:
        p1 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    title=f'BIC_{dimreduct_method}_{clustering_method}',
                    tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), 
                           SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
                    )
        p2 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    title=f'AIC_{dimreduct_method}_{clustering_method}',
                    tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), 
                           SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
                    )
        p3 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    title=f'Silhouette score_{dimreduct_method}_{clustering_method}',
                    tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), 
                           SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
                    )
        
        if all_runs == True:
            p1.line(x=x_axis, y=dataset_10000[: , 0], legend_label = 'num of instances: 10000', 
                    line_width=2, line_color='blue')
            p1.line(x=x_axis, y=dataset_40000[: , 0], legend_label = 'num of instances: 40000', 
                    line_width=2, line_color='red')
            p1.line(x=x_axis, y=dataset_70000[: , 0], legend_label = 'num of instances: 70000', 
                    line_width=2, line_color='green')
        p1.line(x=x_axis, y=dataset_100000[: , 0], legend_label = 'num of instances: 100000', 
                line_width=2, line_color='grey')
        
        if all_runs == True:
            p2.line(x=x_axis, y=dataset_10000[: , 1], legend_label = 'num of instances: 10000', 
                    line_width=2, line_color='blue')
            p2.line(x=x_axis, y=dataset_40000[: , 1], legend_label = 'num of instances: 40000', 
                    line_width=2, line_color='red')
            p2.line(x=x_axis, y=dataset_70000[: , 1], legend_label = 'num of instances: 70000', 
                    line_width=2, line_color='green')
        p2.line(x=x_axis, y=dataset_100000[: , 1], legend_label = 'num of instances: 100000', 
                line_width=2, line_color='grey')
        
        if all_runs == True:
            p3.line(x=x_axis, y=dataset_10000[: , 2], legend_label = 'num of instances: 10000', 
                    line_width=2, line_color='blue')
            p3.line(x=x_axis, y=dataset_40000[: , 2], legend_label = 'num of instances: 40000', 
                    line_width=2, line_color='red')
            p3.line(x=x_axis, y=dataset_70000[: , 2], legend_label = 'num of instances: 70000', 
                    line_width=2, line_color='green')
        p3.line(x=x_axis, y=dataset_100000[: , 2], legend_label = 'num of instances: 100000', 
                line_width=2, line_color='grey')
        
        output_notebook()
        show(row(p1,p2,p3))
                            
        if save_dir != None:
            export_png(p1, filename=f'{save_dir}/BIC_{dimreduct_method}_{clustering_method}.png')
            export_png(p2, filename=f'{save_dir}/AIC_{dimreduct_method}_{clustering_method}.png')
            export_png(p3, filename=f'{save_dir}/Silhouette_score_{dimreduct_method}_{clustering_method}.png')
        
    elif len(dataset_100000[0]) == 1:
        tags = np.load(f'{read_dir}/{dimreduct_method}_{clustering_method}_100000_tags.npy')
        source=ColumnDataSource(dict(tags=tags, x_axis=x_axis, dataset_arr=dataset_100000[: , 0]))
        p = figure(x_axis_label='Sequence of data points',
                   y_axis_label='Benchmark',
                   tools=[BoxZoomTool(), PanTool(), ResetTool(), 
                          SaveTool(), WheelZoomTool(), 
                          HoverTool(tooltips=[('index', '$index'),
                                              ('eps,min clus size,min sample', '@tags')
                                              ]
                                    )
                          ]
                   )
        p.line('x_axis', 'dataset_arr', legend_label = 'Silhouette Score', source = source, line_width = 2)
        output_notebook()
        show(p)
        
def rmsd_heatmap(rmsd_array, title, n, save=False):
    """
    Plot heatmap for a given RMSD array.

    Parameters
    ----------
    rmsd_array : TYPE Numpy Array, shape=(num_of_clusters*n, num_of_clusters*n)
        DESCRIPTION. The pairwise-rmsd array
    title : TYPE String
        DESCRIPTION. The title for the plot
    n : TYPE Integer
        DESCRIPTION. Number of core instances in each cluster
    save : TYPE Bool, optional
        DESCRIPTION Specify whether to save the fig to disk. The default is None.

    Returns
    -------
    None.
    Prints (OR saves) the heat map on screen

    """
    num_clusters=len(rmsd_array)//n
    plt.figure(figsize=(9,9))
    plt.imshow(rmsd_array, cmap='viridis', vmin=0, vmax=1)
    plt.xlabel(f'Frame # (total number of clusters: {num_clusters})')
    plt.ylabel(f'Frame # (total number of clusters: {num_clusters})')
    plt.colorbar(label=r'RMSD ($\AA$)')
    plt.title(title)
    
    if save==False:
        plt.show()
    elif save==True:
        save_dir = '/xspace/hl4212/results/clustering/plots/RMSD_map'
        plt.savefig(f'{save_dir}/{title}.png')
        
def PearsonR_heatmap(dim1_array, dim2_array):
    """
    Given the two Pearson Correlation Coefficient array along different dimensions (PC1 & PC2),
    plot the Pearson R heatmap with Pearson R value shown in plot.

    Parameters
    ----------
    dim1_array : TYPE Numpy array, shape=(num_clusters, num_clusters)
        DESCRIPTION. Pearson R array along 1st dimension
    dim2_array : TYPE Numpy array, shape=(num_clusters, num_clusters)
        DESCRIPTION. Pearson R array along 2nd dimension

    Returns
    -------
    None. Plot the two subplots on screen

    """
    cluster_num = len(dim1_array)
    
    plt.rcParams.update({'font.size': 13, 'font.weight':'bold'})
    fig, ax = plt.subplots(1, 2, figsize=(19,19))
    im_0 = ax[0].imshow(dim1_array, vmin=-1, vmax=1)
    ax[0].grid(False)
    ax[0].xaxis.set(ticks=range(cluster_num))
    ax[0].yaxis.set(ticks=range(cluster_num))
    ax[0].set_title('dim_1')
    ax[0].set_xlabel('Cluster #')
    ax[0].set_ylabel('Cluster #')
    
    im_1 = ax[1].imshow(dim2_array, vmin=-1, vmax=1)
    ax[1].grid(False)
    ax[1].xaxis.set(ticks=range(cluster_num))
    ax[1].yaxis.set(ticks=range(cluster_num))
    ax[1].set_title('dim_2')
    ax[1].set_xlabel('Cluster #')
    ax[1].set_ylabel('Cluster #')
    for i in range(cluster_num):
        for j in range(cluster_num):
            ax[0].text(j, i, format(float(dim1_array[i, j]), '.2f'), ha='center', va='center',
                    color='r')
            ax[1].text(j, i, format(float(dim2_array[i, j]), '.2f'), ha='center', va='center',
                    color='r')
    cbar_0 = ax[0].figure.colorbar(im_0, ax=ax[0], format='% .2f', shrink=0.5)
    cbar_1 = ax[1].figure.colorbar(im_1, ax=ax[1], format='% .2f', shrink=0.5)
    plt.show()

def cluster_population(population_df):
    """
    Plot a vertical bar chart to display cluster populations. X-axis is the 
    cluster index while the height of the bar is the population. Since instances 
    with index -1 does not belong to any clusters, the bar will be colored differently.

    Parameters
    ----------
    population_df : TYPE Pandas DataFrame (containing cluster index & cluster population)
        DESCRIPTION. The dataframe generated by functions calculating cluster 
        population (find out more in Cluster_inspection and High_dim_clustering
        class)

    Returns
    -------
    None. Plot the figure on screen

    """
    # Since instances that has -1 label means they are not in any clusters, color
    # them separately
    color_list = ['firebrick' for _ in range(len(population_df['cluster_idx']))]
    if -1 in list(population_df['cluster_idx']):
        color_list[0] = 'gray'
        
    source = ColumnDataSource(dict(idx=population_df['cluster_idx'],
                                   pop=population_df['population'],
                                   color_list=color_list))
    
    p = figure(title="Cluster population (high dimensional clustering)", x_axis_label='Cluster index',
               y_range=(0, population_df['population'].max()+1000))
    
    p.vbar(x='idx', width=0.7, top='pop', source=source, color='color_list', bottom=0)
    
    # Add box annotation of 5% population
    five_percent_box = BoxAnnotation(top=(population_df['population'].sum()*0.05), bottom=0, fill_alpha=0.1, fill_color='blue')
    p.add_layout(five_percent_box)
    
    # Add labels on top of each bar
    labels = LabelSet(x='idx', y='pop', text='pop', x_offset=-22, y_offset=0, 
                      source=source, render_mode='canvas')
    p.add_layout(labels)
    
    output_notebook()
    show(p)