from typing import Any, Dict, List, Optional, Type, Union
import os
import time
import datetime
import numpy as np

import vtk
# import vtkmodules.all as vtk


vtk_color = {
    # 'Whites': ['antique_white', 'azure', 'bisque', 'blanched_almond',
    #             'cornsilk', 'eggshell', 'floral_white', 'gainsboro',
    #             'ghost_white', 'honeydew', 'ivory', 'lavender',
    #             'lavender_blush', 'lemon_chiffon', 'linen', 'mint_cream',
    #             'misty_rose', 'moccasin', 'navajo_white', 'old_lace',
    #             'papaya_whip', 'peach_puff', 'seashell', 'snow',
    #             'thistle', 'titanium_white', 'wheat', 'white',
    #             'white_smoke', 'zinc_white'],
    'Greys': ['cold_grey', 'dim_grey', 'grey', 'light_grey',
                'slate_grey', 'slate_grey_dark', 'slate_grey_light',
                'warm_grey'],
    'Reds': ['coral', 'coral_light', 
                'hot_pink', 'light_salmon',
                'pink', 'pink_light',
                'raspberry', 'rose_madder', 'salmon',
                ],
    # 'Browns': ['beige', 'brown', 'brown_madder', 'brown_ochre',
    #             'burlywood', 'burnt_sienna', 'burnt_umber', 'chocolate',
    #             'flesh', 'flesh_ochre', 'gold_ochre',
    #             'greenish_umber', 'khaki', 'khaki_dark', 'light_beige',
    #             'peru', 'rosy_brown', 'raw_sienna', 'raw_umber', 'sepia',
    #             'sienna', 'saddle_brown', 'sandy_brown', 'tan',
    #             'van_dyke_brown'],
    'Oranges': ['cadmium_orange', 'cadmium_red_light', 'carrot',
                'dark_orange', 'mars_orange', 'mars_yellow', 'orange',
                'orange_red', 'yellow_ochre'],
    'Yellows': ['aureoline_yellow', 'banana', 'cadmium_lemon',
                'cadmium_yellow', 'cadmium_yellow_light', 'gold',
                'goldenrod', 'goldenrod_dark', 'goldenrod_light',
                'goldenrod_pale', 'light_goldenrod', 'melon',
                'yellow', 'yellow_light'],
    'Greens': ['chartreuse', 'chrome_oxide_green', 'cinnabar_green',
                'cobalt_green', 'emerald_green', 'forest_green', 
                'green_dark', 'green_pale', 'green_yellow', 'lawn_green',
                'lime_green', 'mint', 'olive', 'olive_drab',
                'olive_green_dark', 'permanent_green', 'sap_green',
                'sea_green', 'sea_green_dark', 'sea_green_medium',
                'sea_green_light', 'spring_green', 'spring_green_medium',
                'terre_verte', 'viridian_light', 'yellow_green'],
    'Cyans': ['aquamarine', 'aquamarine_medium', 'cyan', 'cyan_white',
                'turquoise', 'turquoise_dark', 'turquoise_medium',
                'turquoise_pale'],
    'Blues': ['alice_blue', 'blue_light', 'blue_medium',
                'cadet', 'cobalt', 'cornflower', 'cerulean', 'dodger_blue',
                'indigo', 'manganese_blue', 'midnight_blue', 'navy',
                'peacock', 'powder_blue', 'royal_blue', 'slate_blue',
                'slate_blue_dark', 'slate_blue_light',
                'slate_blue_medium', 'sky_blue', 
                'sky_blue_light', 'steel_blue', 'steel_blue_light',
                'turquoise_blue', 'ultramarine'],
    'Magentas': ['blue_violet', 'magenta',
                    'orchid', 'orchid_dark', 'orchid_medium',
                    'plum', 'purple',
                    'purple_medium', 'ultramarine_violet', 'violet',
                    'violet_dark', 'violet_red_medium',
                    'violet_red_pale']
}
color_key = list(vtk_color.keys())


class VTKRender:
    def __init__(
            self, 
            container_size: List[int], 
            win_size: List[int]=[600, 600], 
            offscreen: bool=True,
            auto_render: bool=True
        ) -> None:
        self.container_size = container_size
        self.item_idx = 0
        self.auto_render = auto_render

        # 1. render
        self.render = vtk.vtkRenderer()    
        self.render.SetBackground(1.0, 1.0, 1.0)

        # 2. render window
        self.render_window = vtk.vtkRenderWindow()
        # if offscreen:
        #     self.render_window.SetOffScreenRendering(1)
        self.render_window.SetWindowName("Packing Visualization")
        self.render_window.SetSize(win_size[0], win_size[1])
        self.render_window.AddRenderer(self.render)

        # 3. interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # 4. camera
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(
            2.5 * max(self.container_size), 
            -2 * max(self.container_size), 
            2 * max(self.container_size)
        )
        self.camera.SetViewUp(0, 0, 1.5 * max(container_size))
        self.render.SetActiveCamera(self.camera)

        # 5. axes
        self._init_axes()

        # 6. container (cube)
        self._init_container()

        self.interactor.Initialize()
        self.render_window.Render()

        self.box_id_to_color = {}
        self.used_colors = set()

        self.all_colors = []
        for cat in vtk_color:
            for shade in vtk_color[cat]:
                self.all_colors.append(shade)
        # time.sleep(0.5)
        
        self.frame_index = 0


    def _init_axes(self) -> None:
        axes = vtk.vtkAxesActor()

        transform = vtk.vtkTransform()
        transform.Translate(
            -0.5 * self.container_size[0], 
            -0.5 * self.container_size[1], 
            -0.5 * self.container_size[2]
        )
        
        axes.SetUserTransform(transform)

        sigma = 0.1
        axes_l_x = self.container_size[0] + sigma * self.container_size[2]
        axes_l_y = self.container_size[1] + sigma * self.container_size[2]
        axes_l_z = (1 + sigma) * self.container_size[2]
        
        axes.SetTotalLength(axes_l_x, axes_l_y, axes_l_z)
        axes.SetNormalizedShaftLength(1, 1, 1)
        axes.SetNormalizedTipLength(0.05, 0.05, 0.05)
        axes.AxisLabelsOff()

        self.render.AddActor(axes)
    
    def _init_container(self) -> None:
        container = vtk.vtkCubeSource()
        container.SetXLength(self.container_size[0])
        container.SetYLength(self.container_size[1])
        container.SetZLength(self.container_size[2])
        container.SetCenter([0, 0, 0])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(container.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetRepresentationToWireframe()
        
        self.render.AddActor(actor)
        

    def add_item(self, item_size: List[int], item_pos: List[int], dir: str="") -> None:

        item = vtk.vtkCubeSource()
        item.SetXLength(item_size[0])
        item.SetYLength(item_size[1])
        item.SetZLength(item_size[2])
        item.SetCenter([
            -0.5 * self.container_size[0] + 0.5 * item_size[0] + item_pos[0],
            -0.5 * self.container_size[1] + 0.5 * item_size[1] + item_pos[1],
            -0.5 * self.container_size[2] + 0.5 * item_size[2] + item_pos[2]
        ])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(item.GetOutputPort())
        
        colors = vtk.vtkNamedColors()
        color_0 = color_key[self.item_idx % len(color_key)]
        color_1 = int(self.item_idx / len(color_key))

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("red"))
        actor.GetProperty().EdgeVisibilityOn()
        
        self.render.AddActor(actor)
        # time.sleep(0.5)
        self.render_window.Render()
        
        # time.sleep(0.3)
        actor.GetProperty().SetColor(colors.GetColor3d(vtk_color[color_0][color_1]))
        self.render_window.Render()
        
        self.item_idx += 1

        if not self.auto_render:
            self.hold_on()
    
    def hold_on(self) -> None:
        self.interactor.Start()

    def save_img(self, save_img_path, test_index = None) -> None:

        img_name = str(test_index) + r".png"
        path = save_img_path

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.path.join(path, img_name))
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        
    def save_video_frame(self, save_img_path: str) -> None:
        # Ensure the save directory exists
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        # Save the current frame as an image
        img_name = f"frame_{self.frame_index:04d}.png"  # Sequential frame naming
        img_path = os.path.join(save_img_path, img_name)

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(img_path)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        self.frame_index += 1

    def combine_frames_to_video(self, save_img_path: str, test_index = None, fps: int = 4) -> None:
        video_path = os.path.join(save_img_path, str(test_index) + "video.mp4")
        command = f"ffmpeg -framerate {fps} -i {save_img_path}/frame_%04d.png -c:v mpeg4 -q:v 5 {video_path}"
        os.system(command)
        print(f"Video saved as {video_path}")

    def render_deform_bin(self, box_id_map_3d: np.ndarray) -> None:
        actors = self.render.GetActors()
        actors.InitTraversal()
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if i > 1: 
                self.render.RemoveActor(actor)

        unique_ids = np.unique(box_id_map_3d)
        unique_ids = unique_ids[unique_ids > 0] 
        
        colors = vtk.vtkNamedColors()

        for box_id in unique_ids:
            color_name = self._get_unique_color(box_id)
            
            x_coords, y_coords, z_coords = np.where(box_id_map_3d == box_id)
            
            append = vtk.vtkAppendPolyData()
            
            for i in range(len(x_coords)):
                x, y, z = x_coords[i], y_coords[i], z_coords[i]
                
                cube = vtk.vtkCubeSource()
                cube.SetXLength(1.0)
                cube.SetYLength(1.0)
                cube.SetZLength(1.0)
                cube.SetCenter([
                    -0.5 * self.container_size[0] + 0.5 + x,
                    -0.5 * self.container_size[1] + 0.5 + y,
                    -0.5 * self.container_size[2] + 0.5 + z
                ])
                cube.Update()
                
                append.AddInputData(cube.GetOutput())
            
            append.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(append.GetOutput())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors.GetColor3d(color_name))
            
            self.render.AddActor(actor)
        
        lines = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        shape = box_id_map_3d.shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    current_id = box_id_map_3d[x,y,z]
                    if current_id <= 0:
                        continue
                    
                    if x+1 < shape[0] and box_id_map_3d[x+1,y,z] > 0 and box_id_map_3d[x+1,y,z] != current_id:
                        p1 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y, -0.5*self.container_size[2]+z)
                        p2 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z)
                        p3 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z+1)
                        p4 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y, -0.5*self.container_size[2]+z+1)
                        
                        line = vtk.vtkPolyLine()
                        line.GetPointIds().SetNumberOfIds(5)
                        line.GetPointIds().SetId(0, p1)
                        line.GetPointIds().SetId(1, p2)
                        line.GetPointIds().SetId(2, p3)
                        line.GetPointIds().SetId(3, p4)
                        line.GetPointIds().SetId(4, p1)
                        cells.InsertNextCell(line)
                    
                    if y+1 < shape[1] and box_id_map_3d[x,y+1,z] > 0 and box_id_map_3d[x,y+1,z] != current_id:
                        p1 = points.InsertNextPoint(-0.5*self.container_size[0]+x, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z)
                        p2 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z)
                        p3 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z+1)
                        p4 = points.InsertNextPoint(-0.5*self.container_size[0]+x, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z+1)
                        
                        line = vtk.vtkPolyLine()
                        line.GetPointIds().SetNumberOfIds(5)
                        line.GetPointIds().SetId(0, p1)
                        line.GetPointIds().SetId(1, p2)
                        line.GetPointIds().SetId(2, p3)
                        line.GetPointIds().SetId(3, p4)
                        line.GetPointIds().SetId(4, p1)
                        cells.InsertNextCell(line)
                    
                    if z+1 < shape[2] and box_id_map_3d[x,y,z+1] > 0 and box_id_map_3d[x,y,z+1] != current_id:
                        p1 = points.InsertNextPoint(-0.5*self.container_size[0]+x, -0.5*self.container_size[1]+y, -0.5*self.container_size[2]+z+1)
                        p2 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y, -0.5*self.container_size[2]+z+1)
                        p3 = points.InsertNextPoint(-0.5*self.container_size[0]+x+1, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z+1)
                        p4 = points.InsertNextPoint(-0.5*self.container_size[0]+x, -0.5*self.container_size[1]+y+1, -0.5*self.container_size[2]+z+1)
                        
                        line = vtk.vtkPolyLine()
                        line.GetPointIds().SetNumberOfIds(5)
                        line.GetPointIds().SetId(0, p1)
                        line.GetPointIds().SetId(1, p2)
                        line.GetPointIds().SetId(2, p3)
                        line.GetPointIds().SetId(3, p4)
                        line.GetPointIds().SetId(4, p1)
                        cells.InsertNextCell(line)
        
        lines.SetPoints(points)
        lines.SetLines(cells)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(lines)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(2)
        
        self.render.AddActor(actor)
        self.render_window.Render()

    def _get_unique_color(self, box_id):
        """Get a unique color that has not been used yet"""
        # Try the default color first
        # If that is already used find an unused color

        if box_id in self.box_id_to_color:
            return self.box_id_to_color[box_id]
        
        
        color_category = color_key[box_id % len(color_key)]
        color_shade = (box_id // len(color_key)) % len(vtk_color[color_category])
        color_name = vtk_color[color_category][color_shade]
        
        if color_name in self.used_colors:
            for color in self.all_colors:
                if color not in self.used_colors:
                    color_name = color
                    break

        self.used_colors.add(color_name)
        self.box_id_to_color[box_id] = color_name
        return color_name
    
# if __name__ == "__main__":
#     render = VTKRender([10, 10, 10])

#     # render.add_item([2, 3, 2], [0, 0, 0])
#     # # render.hold_on()
#     # render.add_item([1, 1, 1], [2, 0, 0])
#     # render.add_item([4, 3, 6], [4, 0, 0])

#     # Create a random 3D box ID map
#     box_id_map_3d = np.random.randint(0, 5, (10, 10, 10))

#     render.render_deform_bin(box_id_map_3d)
    

#     render.hold_on()
