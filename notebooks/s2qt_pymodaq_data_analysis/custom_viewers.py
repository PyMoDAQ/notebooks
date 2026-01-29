from typing import List, Union
import numpy as np

from qtpy import QtWidgets

from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_data.data import DataToExport, DataWithAxes, Axis, DataSource

from pymodaq_utils.enums import enum_checker
from pymodaq_utils.factory import ObjectFactory
from pymodaq_gui.plotting import data_viewers
from pymodaq_gui.plotting.data_viewers.base import ViewerBase, ViewersEnum
from pymodaq_gui.utils import DockArea, Dock
from pymodaq_gui.plotting.data_viewers.viewer import ViewerFactory, ViewerDispatcher

viewer_factory = ViewerFactory()

class CustomViewerDispatcher(ViewerDispatcher):
    """Child class of the ViewerDispatcher from pymodaq_gui.

    Parameters
    ----------
    dockarea: DockArea
    title: str
    """
    def __init__(self, dockarea: DockArea = None, title: str = ''):
        super().__init__(dockarea = dockarea, title = title, next_to_dock = None,
                         direction='right')

    def add_viewer(self, viewer_type: ViewersEnum, dock_viewer=None, dock_name=None,
                   direction='right', n_col=0):
        """
        Modified add_viewer to allow for custom positioning
        
        Parameters
        ----------
        viewer_type: ViewersEnum
        dock_viewer: Dock
        dock_name: str
        direction: str, either 'right', 'left', 'bottom', 'top', 'above', 'below'
        n_col: int, default 0, number of columns
               if 0 we do not fine tune the position of the new docks
        """
        viewer_type = enum_checker(ViewersEnum, viewer_type)

        if dock_viewer is None:
            if dock_name is None:
                dock_name = f'{self._title}_Viewer_{len(self.viewer_docks) + 1}'
            dock_viewer = Dock(dock_name, size=(350, 350), closable=False)
        self.viewer_docks.append(dock_viewer)

        self._viewer_widgets.append(QtWidgets.QWidget())
        self.viewers.append(viewer_factory.get(viewer_type.name, parent=self._viewer_widgets[-1]))

        self.viewer_types.append(viewer_type)

        self.viewer_docks[-1].addWidget(self._viewer_widgets[-1])

        i = len(self.viewer_docks)
        if n_col > 0 and i > n_col:
            self.dockarea.addDock(self.viewer_docks[-1], position=direction,
                                  relativeTo=self.viewer_docks[i-n_col-1].name())
        else:
            self.dockarea.addDock(self.viewer_docks[-1], direction)

        
    def update_viewers(self, viewers_type: List[Union[str, ViewersEnum]],
                       viewers_name: List[str] = None, force=False):
        """

        Parameters
        ----------
        viewers_type: List[ViewersEnum]
        viewers_name: List[str] or None
        force: bool
            if True remove all viewers before update else check if new
            viewers type are compatible with old ones

        Returns
        -------
        """
        Nviewers_to_leave = 0

        if not force:
            # check if viewers are compatible with new data dim
            for ind, viewer_type in enumerate(viewers_type):
                if len(self.viewer_types) > ind:
                    if viewer_type == self.viewer_types[ind]:
                        Nviewers_to_leave += 1
                    else:
                        break
                else:
                    break
        self.remove_viewers(Nviewers_to_leave)
        
        ind_loop = 0

        if force:
            Nviewers = len(viewers_type)
            n_col = int(np.ceil(np.sqrt(Nviewers)))
        
            while len(self.viewers) < len(viewers_type):
                direction = 'right'
                if Nviewers > 1  and ind_loop > n_col-1:
                    direction = 'bottom'
                    
                self.add_viewer(viewers_type[ind_loop],
                                dock_name=viewers_name[ind_loop]
                                          if viewers_name is not None else None,
                                direction=direction,
                                n_col=n_col)
                ind_loop += 1

        else:
            while len(self.viewers) < len(viewers_type):
                self.add_viewer(viewers_type[Nviewers_to_leave + ind_loop],
                                dock_name=viewers_name[Nviewers_to_leave + ind_loop]
                                if viewers_name is not None else None,
                                direction=self._direction)
                ind_loop += 1
                
        QtWidgets.QApplication.processEvents()

    def show_data(self, data: DataToExport, force=True):
        """ Convenience method. Display each dwa in a dedicated data viewer"""
        viewer_types = [ViewersEnum.get_viewers_enum_from_data(dwa) for dwa in data]
        viewer_names = [dwa.name for dwa in data]
        if self.viewer_types != viewer_types:
            self.update_viewers(viewer_types, viewer_names, force=force)
        for viewer, dwa in zip(self.viewers, data):
            if len(dwa.axes) != len(dwa.shape):
                dwa.create_missing_axes()
            viewer.show_data(dwa)
