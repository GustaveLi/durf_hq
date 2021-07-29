from bokeh.plotting import figure, show
from bokeh.models.tools import *
from bokeh.models import ColumnDataSource, ColorBar, Title
from bokeh.layouts import row
from bokeh.io import output_notebook, export_png
from bokeh.transform import linear_cmap
from bokeh.palettes import Plasma256
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
        
def cluster_map(instance_array, label_array, center_array=None, title='Cluster_map',save_dir=None):
    """
    Displays differnet clusters in different colors, highlight the position of cluster centers (if provided)

    Parameters
    ----------
    instance_array : TYPE Numpy array, shape=(num_of_instances,2)
        DESCRIPTION. The dataset (2D) used for clustering
    label_array : TYPE Numpy array, shape=(num_of_instances,)
        DESCRIPTION. Labels generated from clustering algorithms
    center_array : TYPE Numpy array, shape=(num_of_clusters, 2) , optional
        DESCRIPTION. Cluster centers generated from clustering algorithms. Some algorithms don't give cluster centers. The default is None.
    title : TYPE String, optional
        DESCRIPTION. Name for the plot. The default is 'Cluster_map'.
    save_dir : TYPE File path (string), optional
        DESCRIPTION. If specified, the plot will be saved to the given directory. The default is None.

    Returns
    -------
    None. Prints (and saves) the plot

    """
    x = instance_array[:, 0].reshape(len(instance_array), )
    y = instance_array[:, 1].reshape(len(instance_array), )
    source = ColumnDataSource(dict(x=x,y=y, label_array=label_array))
    mapper = linear_cmap(field_name='label_array', palette=Plasma256 ,\
                         low=min(label_array) ,high=max(label_array))
    p = figure(title=title, 
               x_axis_label='dim1', 
               y_axis_label='dim2', 
               tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), \
                      SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')],
               )
    p.circle(x='x', y='y', line_color=mapper, fill_color=mapper, 
             source=source, alpha=0.5,size=3)
    try:
        x_center = center_array[:, 0].reshape(len(center_array), )
        y_center = center_array[:, 1].reshape(len(center_array), )
        p.triangle(x=x_center, y=y_center, line_color='green', fill_color='green', size=10)
    except:
        pass
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
    p.add_layout(color_bar, 'right')
    p.add_layout(Title(text="Cluster #", align="center"), "right")
    output_notebook()
    show(p)
    if save_dir != None:
        export_png(p, filename=f'{save_dir}/{title}.png')
        
def preclustering(x_axis, dataset_arr, tags=None):
    if len(dataset_arr[0]) == 2:
        p1 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    )
        p2 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    )
        p1.line(x=x_axis, y=dataset_arr[: , 0], legend_label = 'Inertia', line_width=2)
        p2.line(x=x_axis, y=dataset_arr[: , 1], legend_label = 'Silhouette Score', line_width=2)
        output_notebook()
        show(row(p1,p2))
        
    elif len(dataset_arr[0]) == 3:
        p1 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    )
        p2 = figure(x_axis_label='Num of clusters',
                    y_axis_label='Benchmark',
                    )
        p1.line(x=x_axis, y=dataset_arr[: , 0], legend_label = 'AIC', 
                line_color='blue', line_width=2)
        p1.line(x=x_axis, y=dataset_arr[: , 1], legend_label = 'BIC', 
                line_color='red', line_width=2)
        p2.line(x=x_axis, y=dataset_arr[: , 2], legend_label = 'Silhouette Score', 
                line_width=2)
        output_notebook()
        show(row(p1,p2))
        
    elif len(dataset_arr[0]) == 1:
        source=ColumnDataSource(dict(tags=tags, x_axis=x_axis, dataset_arr=dataset_arr[: , 0]))
        p = figure(x_axis_label='Sequence of data points',
                   y_axis_label='Benchmark',
                   tools=[BoxZoomTool(), PanTool(), ResetTool(), 
                          SaveTool(), WheelZoomTool(), 
                          HoverTool(tooltips=[('data y', '$y'),
                                              ('eps_min_clus_size_min_sample', '@tags')
                                              ]
                                    )
                          ]
                   )
        p.line('x_axis', 'dataset_arr', legend_label = 'Silhouette Score', source = source,
               line_width=2)
        output_notebook()
        show(p)
