# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Visualization tools for `sigima` interactive tests (based on PlotPy)
"""

from __future__ import annotations

import os
from typing import Generator, Literal

import numpy as np
import plotpy.tools
from guidata.qthelpers import exec_dialog as guidata_exec_dialog
from plotpy.builder import make
from plotpy.items import (
    AnnotatedPolygon,
    AnnotatedXRange,
    AnnotatedYRange,
    CurveItem,
    ImageItem,
    LabelItem,
    Marker,
)
from plotpy.plot import (
    BasePlot,
    BasePlotOptions,
    PlotDialog,
    PlotOptions,
    SyncPlotDialog,
)
from plotpy.styles import LINESTYLES, ShapeParam
from qtpy import QtWidgets as QW

from sigima.config import _
from sigima.objects import ImageObj, SignalObj
from sigima.objects.image import CircularROI, PolygonalROI, RectangularROI
from sigima.tests.helpers import get_default_test_name

QAPP: QW.QApplication | None = None

WIDGETS: list[QW.QWidget] = []


def ensure_qapp() -> QW.QApplication:
    """Ensure that a QApplication instance exists."""
    global QAPP
    if QAPP is None:
        QAPP = QW.QApplication.instance()
        if QAPP is None:
            QAPP = QW.QApplication([])  # type: ignore[assignment]
    return QAPP


def exec_dialog(dlg: QW.QDialog) -> None:
    """Execute a dialog, supporting Sphinx-Gallery scraping."""
    global WIDGETS
    gallery_building = os.getenv("SPHINX_GALLERY_BUILDING")
    if gallery_building:
        dlg.show()
        WIDGETS.append(dlg)
    else:
        guidata_exec_dialog(dlg)


TEST_NB = {}

#: Curve colors
COLORS = (
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
)


def style_generator() -> Generator[tuple[str, str], None, None]:
    """Cycling through curve styles"""
    while True:
        for linestyle in LINESTYLES:
            for color in COLORS:
                yield (color, linestyle)


make.style = style_generator()


def get_name_title(name: str | None, title: str | None) -> tuple[str, str]:
    """Return (default) widget name and title

    Args:
        name: Name of the widget, or None to use a default name
        title: Title of the widget, or None to use a default title

    Returns:
        A tuple (name, title) where:
        - `name` is the widget name, which is either the provided name or a default
        - `title` is the widget title, which is either the provided title or a default
    """
    if name is None:
        TEST_NB[name] = TEST_NB.setdefault(name, 0) + 1
        name = get_default_test_name(f"{TEST_NB[name]:02d}")
    if title is None:
        title = f"{_('Test dialog')} `{name}`"
    return name, title


def create_curve_dialog(
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    size: tuple[int, int] | None = None,
) -> PlotDialog:
    """Create Curve Dialog

    Args:
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        size: Size of the dialog as a tuple (width, height), or None for default size

    Returns:
        A `PlotDialog` instance configured for curve plotting
    """
    name, title = get_name_title(name, title)
    win = PlotDialog(
        edit=False,
        toolbar=True,
        title=title,
        options=PlotOptions(
            type="curve", xlabel=xlabel, ylabel=ylabel, xunit=xunit, yunit=yunit
        ),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(name)
    return win


def create_signal_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str | None = None,
) -> CurveItem:
    """Create a signal segment item

    Args:
        x0: X-coordinate of the start point
        y0: Y-coordinate of the start point
        x1: X-coordinate of the end point
        y1: Y-coordinate of the end point
        label: Label for the segment, or None for no label

    Returns:
        A `CurveItem` representing the signal segment
    """
    item = make.annotated_segment(x0, y0, x1, y1, label, show_computations=False)
    item.label.labelparam.bgalpha = 0.5
    item.label.labelparam.anchor = "T"
    item.label.labelparam.yc = 10
    item.label.labelparam.update_item(item.label)
    p: ShapeParam = item.shape.shapeparam
    p.line.color = "#33ff00"
    p.line.width = 5
    p.symbol.facecolor = "#26be00"
    p.symbol.edgecolor = "#33ff00"
    p.symbol.marker = "Ellipse"
    p.symbol.size = 11
    p.update_item(item.shape)
    item.set_movable(False)
    item.set_resizable(False)
    item.set_selectable(False)
    return item


def create_cursor(
    orientation: Literal["h", "v"], position: float, label: str
) -> Marker:
    """Create a horizontal or vertical cursor item

    Args:
        orientation: 'h' for horizontal cursor, 'v' for vertical cursor
        position: Position of the cursor along the relevant axis
        label: Label format string for the cursor

    Returns:
        A `Marker` representing the cursor
    """
    if orientation == "h":
        cursor = make.hcursor(position, label=label)
    elif orientation == "v":
        cursor = make.vcursor(position, label=label)
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    cursor.set_movable(False)
    cursor.set_selectable(False)
    cursor.markerparam.line.color = "#a7ff33"
    cursor.markerparam.line.width = 3
    cursor.markerparam.symbol.marker = "NoSymbol"
    cursor.markerparam.text.textcolor = "#ffffff"
    cursor.markerparam.text.background_color = "#000000"
    cursor.markerparam.text.background_alpha = 0.5
    cursor.markerparam.text.font.bold = True
    cursor.markerparam.update_item(cursor)
    return cursor


def create_range(
    orientation: Literal["h", "v"], pos_min: float, pos_max: float, title: str
) -> AnnotatedXRange | AnnotatedYRange:
    """Create a horizontal or vertical range item

    Args:
        orientation: 'h' for horizontal range, 'v' for vertical range
        pos_min: Minimum position of the range along the relevant axis
        pos_max: Maximum position of the range along the relevant axis
        title: Title for the range

    Returns:
        An `AnnotatedXRange` or `AnnotatedYRange` representing the range
    """
    if orientation == "h":
        item = make.annotated_xrange(
            pos_min, pos_max, title=title, show_computations=False
        )
    elif orientation == "v":
        item = make.annotated_yrange(
            pos_min, pos_max, title=title, show_computations=False
        )
    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    item.label.labelparam.bgalpha = 0.5
    item.label.labelparam.anchor = "L"
    item.label.labelparam.xc = 20
    item.label.labelparam.update_item(item.label)
    item.set_movable(False)
    item.set_resizable(False)
    item.set_selectable(False)
    return item


def create_label(text: str) -> LabelItem:
    """Create a text label item

    Args:
        text: Text content of the label

    Returns:
        A `LabelItem` representing the text label
    """
    item = make.label(text, "TL", (0, 0), "TL")
    return item


def get_object_name_from_title(title: str, fallback: str) -> str:
    """Generate a valid object name from a title string

    Args:
        title: The title string to convert
        fallback: Fallback name to use if title is empty or invalid

    Returns:
        A valid object name derived from the title or the fallback name
    """
    if title:
        obj_name = "".join(c if c.isalnum() else "_" for c in title)
        if obj_name:
            return obj_name
    return fallback


def view_curve_items(
    items: list[CurveItem],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    add_legend: bool = True,
    object_name: str = "",
) -> None:
    """Create a curve dialog and plot items

    Args:
        items: List of `CurveItem` objects to plot
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        add_legend: Whether to add a legend to the plot, default is True
        object_name: Object name for the dialog (for screenshot functionality)
    """
    ensure_qapp()
    win = create_curve_dialog(
        name=name, title=title, xlabel=xlabel, ylabel=ylabel, xunit=xunit, yunit=yunit
    )
    win.setObjectName(object_name or get_object_name_from_title(title, "curve_dialog"))
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    if add_legend:
        plot.add_item(make.legend())
    exec_dialog(win)
    make.style = style_generator()  # Reset style generator for next call


def view_curves(
    data_or_objs: list[SignalObj | np.ndarray | tuple[np.ndarray, np.ndarray]]
    | SignalObj
    | np.ndarray
    | tuple[np.ndarray, np.ndarray],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    object_name: str = "",
) -> None:
    """Create a curve dialog and plot curves

    Args:
        data_or_objs: Single `SignalObj` or `np.ndarray`, or a list/tuple of these,
         or a list/tuple of (xdata, ydata) pairs
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        object_name: Object name for the dialog (for screenshot functionality)
    """
    ensure_qapp()
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]
    items = []
    curve_title: str | None = None
    for data_or_obj in datalist:
        if isinstance(data_or_obj, SignalObj):
            data = data_or_obj.xydata
            if data_or_obj.title:
                curve_title = data_or_obj.title
            if data_or_obj.xlabel and xlabel is None:
                xlabel = data_or_obj.xlabel
            if data_or_obj.ylabel and ylabel is None:
                ylabel = data_or_obj.ylabel
            if data_or_obj.xunit and xunit is None:
                xunit = data_or_obj.xunit
            if data_or_obj.yunit and yunit is None:
                yunit = data_or_obj.yunit
        elif isinstance(data_or_obj, np.ndarray):
            data = data_or_obj
        elif isinstance(data_or_obj, (tuple, list)) and len(data_or_obj) == 2:
            data = data_or_obj
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")
        if len(data) == 2:
            xdata, ydata = data
            item = make.mcurve(xdata, ydata)
        else:
            item = make.mcurve(data)
        if curve_title is not None:
            item.setTitle(curve_title)
        items.append(item)
    view_curve_items(
        items,
        name=name,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xunit=xunit,
        yunit=yunit,
        object_name=object_name,
    )
    make.style = style_generator()  # Reset style generator for next call


def create_image_dialog(
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    size: tuple[int, int] | None = None,
    object_name: str = "",
) -> PlotDialog:
    """Create Image Dialog

    Args:
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        size: Size of the dialog as a tuple (width, height), or None for default size
        object_name: Object name for the dialog (for screenshot functionality)

    Returns:
        A `PlotDialog` instance configured for image plotting
    """
    ensure_qapp()
    name, title = get_name_title(name, title)
    win = PlotDialog(
        edit=False,
        toolbar=True,
        title=title,
        options=PlotOptions(
            type="image",
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
        ),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(object_name or name)
    for toolklass in (
        plotpy.tools.LabelTool,
        plotpy.tools.VCursorTool,
        plotpy.tools.HCursorTool,
        plotpy.tools.XCursorTool,
        plotpy.tools.AnnotatedRectangleTool,
        plotpy.tools.AnnotatedCircleTool,
        plotpy.tools.AnnotatedEllipseTool,
        plotpy.tools.AnnotatedSegmentTool,
        plotpy.tools.AnnotatedPointTool,
    ):
        win.get_manager().add_tool(toolklass, switch_to_default_tool=True)
    return win


def view_image_items(
    items: list[ImageItem],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    show_itemlist: bool = False,
    object_name: str = "",
) -> None:
    """Create an image dialog and show items

    Args:
        items: List of `ImageItem` objects to display
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        show_itemlist: Whether to show the item list panel in the dialog,
         default is False
        object_name: Object name for the dialog (for screenshot functionality)
    """
    ensure_qapp()
    win = create_image_dialog(
        name=name,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        xunit=xunit,
        yunit=yunit,
        zunit=zunit,
        object_name=object_name,
    )
    if show_itemlist:
        win.manager.get_itemlist_panel().show()
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    exec_dialog(win)


def view_images(
    data_or_objs: list[ImageObj | np.ndarray] | ImageObj | np.ndarray,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    object_name: str = "",
) -> None:
    """Create an image dialog and show images

    Args:
        data_or_objs: Single `ImageObj` or `np.ndarray`, or a list/tuple of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        object_name: Object name for the dialog (for screenshot functionality)
    """
    ensure_qapp()
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]
    items = []
    image_title: str | None = None
    for data_or_obj in datalist:
        if isinstance(data_or_obj, ImageObj):
            data = data_or_obj.data
            if data_or_obj.title:
                image_title = data_or_obj.title
            if data_or_obj.xlabel and xlabel is None:
                xlabel = data_or_obj.xlabel
            if data_or_obj.ylabel and ylabel is None:
                ylabel = data_or_obj.ylabel
            if data_or_obj.zlabel and zlabel is None:
                zlabel = data_or_obj.zlabel
            if data_or_obj.xunit and xunit is None:
                xunit = data_or_obj.xunit
            if data_or_obj.yunit and yunit is None:
                yunit = data_or_obj.yunit
            if data_or_obj.zunit and zunit is None:
                zunit = data_or_obj.zunit
        elif isinstance(data_or_obj, np.ndarray):
            data = data_or_obj
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")
        # Display real and imaginary parts of complex images.
        assert data is not None
        kwargs = {"interpolation": "nearest", "eliminate_outliers": 0.1}
        if np.issubdtype(data.dtype, np.complexfloating):
            re_title = f"Re({image_title})" if image_title is not None else "Real"
            im_title = f"Im({image_title})" if image_title is not None else "Imaginary"
            items.append(make.image(data.real, title=re_title, **kwargs))
            items.append(make.image(data.imag, title=im_title, **kwargs))
        else:
            items.append(make.image(data, title=image_title, **kwargs))
    view_image_items(
        items,
        name=name,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        xunit=xunit,
        yunit=yunit,
        zunit=zunit,
        object_name=object_name,
    )


def view_curves_and_images(
    data_or_objs: list[SignalObj | np.ndarray | ImageObj | np.ndarray],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    zunit: str | None = None,
    object_name: str = "",
) -> None:
    """View signals, then images in two successive dialogs

    Args:
        data_or_objs: List of `SignalObj`, `ImageObj`, `np.ndarray` or a mix of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        zlabel: Label for the z-axis (color scale), or None for no label
        xunit: Unit for the x-axis, or None for no unit
        yunit: Unit for the y-axis, or None for no unit
        zunit: Unit for the z-axis (color scale), or None for no unit
        object_name: Object name for the dialog (for screenshot functionality)
    """
    ensure_qapp()
    if isinstance(data_or_objs, (tuple, list)):
        objs = data_or_objs
    else:
        objs = [data_or_objs]
    sig_objs = [obj for obj in objs if isinstance(obj, (SignalObj, np.ndarray))]
    if sig_objs:
        view_curves(
            sig_objs,
            name=name,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            object_name=f"{object_name}_curves",
        )
    ima_objs = [obj for obj in objs if isinstance(obj, (ImageObj, np.ndarray))]
    if ima_objs:
        view_images(
            ima_objs,
            name=name,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
            object_name=f"{object_name}_images",
        )


def __compute_grid(
    num_objects: int, max_cols: int = 4, fixed_num_rows: int | None = None
) -> tuple[int, int]:
    """Compute number of rows and columns for a grid of images

    Args:
        num_objects: Total number of objects to display
        max_cols: Maximum number of columns in the grid
        fixed_num_rows: Fixed number of rows, if specified

    Returns:
        A tuple (num_rows, num_cols) representing the grid dimensions
    """
    num_cols = min(num_objects, max_cols)
    if fixed_num_rows is not None:
        num_rows = fixed_num_rows
        num_cols = (num_objects + num_rows - 1) // num_rows
    else:
        num_rows = (num_objects + num_cols - 1) // num_cols
    return num_rows, num_cols


def view_images_side_by_side(
    images: list[ImageItem | np.ndarray | ImageObj],
    titles: list[str],
    share_axes: bool = True,
    rows: int | None = None,
    maximized: bool = False,
    title: str | None = None,
    object_name: str = "",
) -> None:
    """Show sequence of images

    Args:
        images: List of `ImageItem`, `np.ndarray`, or `ImageObj` objects to display
        titles: List of titles for each image
        share_axes: Whether to share axes across plots, default is True
        rows: Fixed number of rows in the grid, or None to compute automatically
        maximized: Whether to show the dialog maximized, default is False
        title: Title of the dialog, or None for a default title
        object_name: Object name for the dialog widget (used for screenshot filename)
    """
    ensure_qapp()
    # pylint: disable=too-many-nested-blocks
    rows, cols = __compute_grid(len(images), fixed_num_rows=rows, max_cols=4)
    dlg = SyncPlotDialog(title=title)
    dlg.setObjectName(
        object_name or get_object_name_from_title(title, "images_side_by_side")
    )
    for idx, (img, imtitle) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols
        plot = BasePlot(options=BasePlotOptions(title=imtitle))
        other_items = []
        if isinstance(img, ImageItem):
            item = img
        else:
            if isinstance(img, ImageObj):
                data = img.data
                mask = img.maskdata
                imtitle = img.title or imtitle
            elif isinstance(img, np.ndarray):
                data = img
                mask = np.zeros_like(data, dtype=bool)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            item = make.maskedimage(
                data,
                mask,
                title=imtitle,
                interpolation="nearest",
                colormap="viridis",
                eliminate_outliers=0.1,
                show_mask=True,
            )
            if isinstance(img, ImageObj):
                x0, y0, dx, dy = img.x0, img.y0, img.dx, img.dy
                item.param.xmin, item.param.xmax = x0, x0 + dx * data.shape[1]
                item.param.ymin, item.param.ymax = y0, y0 + dy * data.shape[0]
                item.param.update_item(item)
                if img.roi is not None and not img.roi.is_empty():
                    for single_roi in img.roi:
                        if isinstance(single_roi, RectangularROI):
                            x0, y0, x1, y1 = single_roi.get_bounding_box(img)
                            roi_item = make.annotated_rectangle(
                                x0, y0, x1, y1, single_roi.title
                            )
                        elif isinstance(single_roi, CircularROI):
                            x0, y0, x1, y1 = single_roi.get_bounding_box(img)
                            roi_item = make.annotated_circle(
                                x0, y0, x1, y1, single_roi.title
                            )
                        elif isinstance(single_roi, PolygonalROI):
                            coords = single_roi.get_physical_coords(img)
                            points = np.array(coords).reshape(-1, 2)
                            roi_item = AnnotatedPolygon(points)
                            roi_item.annotationparam.title = single_roi.title
                            roi_item.set_style("plot", "shape/drag")
                            roi_item.annotationparam.update_item(roi_item)
                        other_items.append(roi_item)
        plot.add_item(item)
        for other_item in other_items:
            plot.add_item(other_item)
        dlg.add_plot(row, col, plot, sync=share_axes)
    dlg.finalize_configuration()
    if maximized:
        dlg.resize(1200, 800)
        dlg.showMaximized()
    exec_dialog(dlg)
