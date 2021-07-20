from bokeh.plotting import figure, show
from bokeh.models.tools import *
from bokeh.models import ColumnDataSource, ColorBar, Title
from bokeh.io import output_notebook, export_png
from bokeh.transform import linear_cmap
from bokeh.palettes import Plasma256
import mdtraj as md

file_dir = '/xspace/hl4212/DURF_datasets/triad_molecule'
traj_path = f'{file_dir}/triad_dataset_aligned.nc'
top_path = f'{file_dir}/triad_forcefield_ground.prmtop'
traj=md.load(traj_path, top=top_path)

def RMSD_map(array, title='RMSD_map', save_dir=None, ref_frame=0):
    """
     Visualize the results from given 2D array, color each point with rmsd value

    Parameters
    ----------
    title : TYPE Str, optional
        DESCRIPTION. Specify the title of the plot. The default is 'RMSD_map'.
    array : TYPE Numpy array, shape=(num of instances, 2)
        DESCRIPTION. The array used for plotting
    save_dir : TYPE file path (string), optional
        DESCRIPTION. If specified, the plot will be saved to the given directory. The default is None.
    ref_frame : TYPE int, optional
        DESCRIPTION. The reference frame for RMSD calculation. The default is 0.

    Returns
    -------
    None. Only prints (and saves) the plot

    """
    # check for dimensionality
    if array.shape[-1]==2:        
        x = array[:, 0].reshape(len(array), )
        y = array[:, 1].reshape(len(array), )
        rmsd_array = md.rmsd(traj, traj, frame=ref_frame)
        source = ColumnDataSource(dict(x=x,y=y, rmsd_array=rmsd_array))
        mapper = linear_cmap(field_name='rmsd_array', palette=Plasma256 ,low=min(rmsd_array) ,high=max(rmsd_array))
        p = figure(title=title, 
                   x_axis_label='dim1', 
                   y_axis_label='dim2', 
                   tools=[BoxZoomTool(), HoverTool(), PanTool(), ResetTool(), SaveTool(), WheelZoomTool(), BoxSelectTool(mode='append')]
                  )
        p.circle(x='x', y='y', line_color=mapper, fill_color=mapper, source=source, size=3)
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
        p.add_layout(color_bar, 'right')
        p.add_layout(Title(text="RMSD", align="center"), "right")
        output_notebook()
        show(p)
        if save_dir != None:
            export_png(p, filename=f'{save_dir}/{title}.png')
    else:
        print('Sorry, only dimension reduced to 2D accepted.')