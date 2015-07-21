#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstrate the mismatch issue of implicit rendering
"""

import numpy as np
import vtk
from vtk.util import numpy_support

# ===========================================================================
# Initialisation
# coef = [.5, 1, .2, 0, .1, 0, 0, .2, 0, 0]
coef = np.random.standard_normal(10)
contour_value = 0.1


def get_actor(vtk_source, color=[1, 1, 0]):

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(vtk_source.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.ReleaseDataFlagOn()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(normals.GetOutputPort())
    stripper.ReleaseDataFlagOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.SetScalarVisibility(0)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(color)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)

    return actor


def vtk_show(_renderer_1, _renderer_2, width=640 * 2, height=480):

    # Multiple Viewports
    xmins = [0.0, 0.5]
    xmaxs = [0.5, 1.0]
    ymins = [0.0, 0.0]
    ymaxs = [1.0, 1.0]

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(_renderer_1)
    render_window.AddRenderer(_renderer_2)
    _renderer_1.ResetCamera()
    _renderer_2.ResetCamera()

    _renderer_1.SetViewport(xmins[0], ymins[0], xmaxs[0], ymaxs[0])
    _renderer_2.SetViewport(xmins[1], ymins[1], xmaxs[1], ymaxs[1])

    render_window.SetSize(width, height)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(interactor_style)

    _renderer_2.SetActiveCamera(_renderer_1.GetActiveCamera())

    # Add a x-y-z coordinate to the original point
    axes_coor = vtk.vtkAxes()
    axes_coor.SetOrigin(0, 0, 0)
    mapper_axes_coor = vtk.vtkPolyDataMapper()
    mapper_axes_coor.SetInputConnection(axes_coor.GetOutputPort())
    actor_axes_coor = vtk.vtkActor()
    actor_axes_coor.SetMapper(mapper_axes_coor)
    renderer_1.AddActor(actor_axes_coor)
    renderer_2.AddActor(actor_axes_coor)

    iren.Initialize()
    iren.Start()

# ===========================================================================
# Left view: Quadric defined with vtkQuadric
quadric_1 = vtk.vtkQuadric()
quadric_1.SetCoefficients(coef)

sample_1 = vtk.vtkSampleFunction()
sample_1.SetImplicitFunction(quadric_1)
sample_1.ComputeNormalsOff()

contour_1 = vtk.vtkContourFilter()
contour_1.SetInputConnection(sample_1.GetOutputPort())
contour_1.SetValue(0, contour_value)

# ===========================================================================
# Right view: Quadric defined with meshgrid
range_x = [-1, 1]
range_y = [-1, 1]
range_z = [-1, 1]
step = 0.05
offset = 0.1
step_x = np.arange(range_x[0] - offset, range_x[1] + offset, step)
step_y = np.arange(range_y[0] - offset, range_y[1] + offset, step)
step_z = np.arange(range_z[0] - offset, range_z[1] + offset, step)
[x, y, z] = np.meshgrid(step_x, step_y, step_z)
dim_x = len(x)
dim_y = len(y)
dim_z = len(z)

quadric_2 = coef[0] * x * x + coef[1] * y * y + coef[2] * z * z + \
    coef[3] * x * y + coef[4] * y * z + coef[5] * x * z + \
    coef[6] * x + coef[7] * y + coef[8] * z + \
    coef[9] * np.ones((x.shape))

# Convert numpy array to vtkFloatArray
vtk_array = numpy_support.numpy_to_vtk(
    num_array=quadric_2.transpose(2, 0, 1).ravel(),  # was (2, 1, 0)
    deep=True,
    array_type=vtk.VTK_FLOAT)

spacing = [(range_x[1] - range_x[0]) / (dim_x - 1.0),
           (range_y[1] - range_y[0]) / (dim_y - 1.0),
           (range_z[1] - range_z[0]) / (dim_z - 1.0)]

# Convert vtkFloatArray to vtkImageData
vtk_image_data = vtk.vtkImageData()
vtk_image_data.SetDimensions(quadric_2.shape)
vtk_image_data.SetSpacing([0.1] * 3)  # How to set a correct spacing value??
vtk_image_data.GetPointData().SetScalars(vtk_array)
vtk_image_data.SetOrigin(-1, -1, -1)
vtk_image_data.SetSpacing(spacing)

dims = vtk_image_data.GetDimensions()
bounds = vtk_image_data.GetBounds()


implicit_volume = vtk.vtkImplicitVolume()
implicit_volume.SetVolume(vtk_image_data)

sample_2 = vtk.vtkSampleFunction()
sample_2.SetImplicitFunction(implicit_volume)
sample_2.SetModelBounds(bounds)
sample_2.ComputeNormalsOff()

contour_2 = vtk.vtkContourFilter()
contour_2.SetInputConnection(sample_2.GetOutputPort())
contour_2.SetValue(0, contour_value)

# ===========================================================================
# Rendering
renderer_1 = vtk.vtkRenderer()  # for vtkQuadric
renderer_2 = vtk.vtkRenderer()  # for meshgrid

actor_1 = get_actor(contour_1)
acotr_2 = get_actor(contour_2)

renderer_1.AddActor(actor_1)
renderer_2.AddActor(acotr_2)

vtk_show(renderer_1, renderer_2)
