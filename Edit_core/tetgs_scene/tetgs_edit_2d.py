import torch.nn as nn
import open3d as o3d
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.spherical_harmonics import (
    eval_sh, RGB2SH
)
from utils.graphics_utils import *
from utils.general_utils import inverse_sigmoid
from tetgs_scene.gs_model import GaussianSplattingWrapper
from tetgs_scene.cameras import CamerasWrapper

scale_activation = torch.exp
scale_inverse_activation = torch.log
use_old_method = False


class EditTetGS(nn.Module):
    def __init__(
        self, 
        nerfmodel: GaussianSplattingWrapper,
        points: torch.Tensor,
        colors: torch.Tensor,
        keep_gaussians,  # reconstructed gaussian attrs
        keep_faces_num:int,  # reconstructed mesh faces
        keep_vertices_num:int, # reconstructed mesh vertices
        nerf_cameras: CamerasWrapper=None,
        initialize:bool=True,
        sh_levels:int=4,
        learnable_positions:bool=True,
        learn_color_only=False,
        freeze_gaussians=False,
        surface_mesh_to_bind=None,  # Open3D mesh
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=False,
        learn_surface_mesh_opacity=False,
        learn_surface_mesh_scales=False,
        *args, **kwargs) -> None:
        
        super(EditTetGS, self).__init__()
        
        self.nerfmodel = nerfmodel
        self.freeze_gaussians = freeze_gaussians
        
        if nerf_cameras is None:
            self.nerf_cameras = nerfmodel.training_cameras
        else:
            self.nerf_cameras = nerf_cameras
        
        self.learn_positions = ((not learn_color_only) and learnable_positions) and (not freeze_gaussians)
        self.learn_opacities = (not learn_color_only) and (not freeze_gaussians)
        self.learn_scales = (not learn_color_only) and (not freeze_gaussians)
        self.learn_quaternions = (not learn_color_only) and (not freeze_gaussians)
        self.learnable_positions = learnable_positions
        
        if surface_mesh_to_bind is not None:
            self.binded_to_surface_mesh = True
            self.bind_3dgs = False
            self.fixed_2dgs = True
            self.learn_positions = learn_surface_mesh_positions
            self.learn_scales = learn_surface_mesh_scales
            self.learn_quaternions = learn_surface_mesh_scales
            self.learn_opacities = learn_surface_mesh_opacity
            
            self._surface_mesh_faces = torch.nn.Parameter(
                torch.tensor(np.array(surface_mesh_to_bind.triangles)).to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            self._verts_points = torch.nn.Parameter(
                torch.tensor(np.array(surface_mesh_to_bind.vertices)).float().to(nerfmodel.device),
                requires_grad=False).to(nerfmodel.device)
            # thickness
            if surface_mesh_thickness is None:
                surface_mesh_thickness = self.nerf_cameras.get_spatial_extent() / 1_000_000
            self.surface_mesh_thickness = torch.nn.Parameter(
                torch.tensor(surface_mesh_thickness).to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            
            # ================ extract edit mesh ==================
            # all faces num
            all_faces = torch.tensor(np.array(surface_mesh_to_bind.triangles)).to(nerfmodel.device)
            self.all_faces_num = all_faces.shape[0]
            # all vertices num
            all_vertices = torch.tensor(np.array(surface_mesh_to_bind.vertices)).float().to(nerfmodel.device)
            self.all_vertices_num = all_vertices.shape[0]
            # TODO: fix
            self.keep_faces_num = keep_faces_num  # keep faces num
            self.keep_vertices_num = keep_vertices_num # keep vertices num
            self.edit_faces_num = self.all_faces_num - self.keep_faces_num  # edit faces num
            self.edit_vertices_num = self.all_vertices_num - self.keep_vertices_num  # edit vertices num
            # TODO: fix
            # edit_mesh faces & verts
            edit_mesh_faces = all_faces[self.keep_faces_num:, :] - self.keep_vertices_num
            self._edit_mesh_faces = torch.nn.Parameter(
                edit_mesh_faces,
                requires_grad=False).to(nerfmodel.device)
            edit_mesh_vertices = all_vertices[self.keep_vertices_num:, :]
            self._edit_mesh_vertices = torch.nn.Parameter(
                edit_mesh_vertices,
                requires_grad=False).to(nerfmodel.device)
            print("edit_mesh_vertices shape: ", self._edit_mesh_vertices.shape)
            
            # edit verts color
            num_vertices = len(edit_mesh_vertices)
            gray_colors = np.ones((num_vertices, 3)) * 0.5
            
            self._vertex_colors = torch.tensor(gray_colors).float().to(nerfmodel.device)
            print("self._vertex_colors shape: ", self._vertex_colors.shape)
            
            # construct edit_mesh
            self._edit_surface_mesh = Meshes(
                verts=[edit_mesh_vertices.to(self.device)],
                faces=[edit_mesh_faces.to(self.device)],
                textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            )
    
            edit_faces_color = self._vertex_colors[self._edit_mesh_faces]  # n_faces, 3, n_coords
            edit_colors, _ = self.calculate_attr_by_bary_coords(edit_faces_color[:, None]) # [n_edit_gaussians, 3]
            
            # position
            self._points_mesh = edit_mesh_vertices
            # First gather vertices of all triangles
            faces_verts = self._points_mesh[self._edit_mesh_faces]
            # edit gaussian points and face_indices
            edit_points, edit_face_indices = self.calculate_attr_by_bary_coords(faces_verts[:, None])
            self._edit_points = nn.Parameter(
                edit_points,
                requires_grad=self.learn_positions   # False
            ).to(nerfmodel.device)
            self._edit_face_indices = nn.Parameter(
                edit_face_indices,
                requires_grad=False
            ).to(nerfmodel.device)
            n_edit_points = edit_points.shape[0]
            self._n_edit_points = n_edit_points
        else:
            raise NotImplementedError("TetGS must be binded to a mesh")
        
        # Render parameters
        self.image_height = int(self.nerf_cameras.height[0].item())
        self.image_width = int(self.nerf_cameras.width[0].item())
        self.focal_factor = max(self.nerf_cameras.p3d_cameras.K[0, 0, 0].item(),
                                self.nerf_cameras.p3d_cameras.K[0, 1, 1].item())
        self.fx = self.nerf_cameras.fx
        self.fy = self.nerf_cameras.fy
        
        # denisty
        if self.binded_to_surface_mesh and (not self.learn_opacities):
            all_edit_densities = inverse_sigmoid(0.9999 * torch.ones((n_edit_points, 1), dtype=torch.float, device=edit_points.device))
            self.learn_opacities = False
        else:
            all_edit_densities = inverse_sigmoid(0.1 * torch.ones((n_edit_points, 1), dtype=torch.float, device=edit_points.device))
        self.all_edit_densities = nn.Parameter(
            all_edit_densities,
            requires_grad=self.learn_opacities
        ).to(nerfmodel.device)
        self.return_one_densities = False
         
        self.min_ndc_radius = 2. / min(self.image_height, self.image_width)
        self.max_ndc_radius = 2. * 0.01  # 2. * 0.01
        self.min_radius = None # self.min_ndc_radius / self.focal_factor * 0.005  # 0.005
        self.max_radius = None # self.max_ndc_radius / self.focal_factor * 2.  # 2.
        self.radius_dim = 7
        
        # Initialize learnable radiuses
        if self.binded_to_surface_mesh and self.fixed_2dgs:
            self.scale_activation = scale_activation
            self.scale_inverse_activation = scale_inverse_activation

            # only for edit mesh
            faces_verts = edit_mesh_vertices[self._edit_mesh_faces]   # [M, 3, 3]
            triangles = faces_verts[self._edit_face_indices[:, 0], :]  # [n_edit_points, 3, 3]
            eps = 1e-8
            
            normals = torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=1
            )
            v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
            v1 = triangles[:, 1] - triangles[:, 0]
            v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
            v1 = v1 / v1_norm
            v2 = torch.cross(v0, v1, dim=-1)
            v2_norm = torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps
            v2 = v2 / v2_norm
            # stack后, v是行向量
            rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
            rotation = rotation.view(-1, 3, 3)
            # 3dgs rotation中的v是列向量, 要转置
            rotation = rotation.transpose(-2, -1)
            edit_quaternions = matrix_to_quaternion(rotation)
            self._edit_quaternions = nn.Parameter(
                edit_quaternions,
                requires_grad=self.learn_quaternions
            ).to(nerfmodel.device)
            
            # min dist as circle radius
            distances = calculate_distances(self._edit_points, triangles[:, 0], triangles[:, 1], triangles[:, 2])
            s1 = distances
            s2 = distances
            s0 = eps * torch.ones_like(s1)
            edit_scales = torch.concat((s0, s1, s2), dim=1).reshape(-1, 3)
            self._edit_scales = nn.Parameter(
                scale_inverse_activation(edit_scales),
                requires_grad=self.learn_scales
            ).to(nerfmodel.device)
        else:
            raise NotImplementedError
        
        # Initialize color features
        self.edit_sh_levels = sh_levels
        edit_sh_coordinates_dc = RGB2SH(edit_colors).unsqueeze(dim=1)
        self._edit_sh_coordinates_dc = nn.Parameter(
            edit_sh_coordinates_dc.to(self.nerfmodel.device),
            requires_grad=True and (not freeze_gaussians)
        ).to(self.nerfmodel.device)
        if sh_levels > 1:
            self._edit_sh_coordinates_rest = nn.Parameter(
                torch.zeros(n_edit_points, sh_levels**2 - 1, 3).to(self.nerfmodel.device),
                requires_grad=True and (not freeze_gaussians)
            ).to(self.nerfmodel.device)

        
        # =========== load reconstruction gaussian attrs ============
        keep_points = keep_gaussians["keep_xyz"].to(nerfmodel.device)
        keep_opacities = keep_gaussians["keep_opacities"].to(nerfmodel.device)
        keep_scales = keep_gaussians["keep_scales"].to(nerfmodel.device)
        keep_quaternions = keep_gaussians["keep_rots"].to(nerfmodel.device)
        keep_sh_coordinates_dc = keep_gaussians["keep_sh_coordinates_dc"].to(nerfmodel.device)
        keep_face_indices = keep_gaussians["keep_face_indices"].to(nerfmodel.device)
        keep_sh_levels = keep_gaussians["sh_level"]
        if keep_sh_levels > 1:
            keep_sh_coordinates_rest = keep_gaussians["keep_sh_coordinates_rest"].to(nerfmodel.device)
        
        # recon attrs are not learnable
        self._keep_points = nn.Parameter(
            keep_points,
            requires_grad=False   
        ).to(self.nerfmodel.device)
        self.all_keep_densities = nn.Parameter(
            keep_opacities,
            requires_grad=False   
        ).to(self.nerfmodel.device)
        self._keep_scales = nn.Parameter(
            keep_scales,
            requires_grad=False   
        ).to(self.nerfmodel.device)
        self._keep_quaternions = nn.Parameter(
            keep_quaternions,
            requires_grad=False   
        ).to(self.nerfmodel.device)
        self._keep_sh_coordinates_dc = nn.Parameter(
            keep_sh_coordinates_dc,
            requires_grad=False   
        ).to(self.nerfmodel.device)
        self.keep_sh_levels = keep_sh_levels
        if keep_sh_levels > 1:
            self._keep_sh_coordinates_rest = nn.Parameter(
                keep_sh_coordinates_rest,
                requires_grad=False   
            ).to(self.nerfmodel.device)
        self._keep_face_indices = nn.Parameter(
            keep_face_indices,
            requires_grad=False   
        ).to(self.nerfmodel.device)
        self._n_keep_points = self._keep_points.shape[0]
    
    @property
    def device(self):
        return self.nerfmodel.device
    
    @property
    def n_points(self):
        _n_points = self._n_edit_points + self._n_keep_points
        return _n_points
    
    @property
    def points(self):
        full_points = torch.cat([self._keep_points, self._edit_points], dim=0)
        return full_points
            
    @property
    def strengths(self):
        all_densities = torch.cat([self.all_keep_densities, self.all_edit_densities], dim=0)
        if self.return_one_densities:
            return torch.ones_like(all_densities.view(-1, 1))
        else:
            return torch.sigmoid(all_densities.view(-1, 1))
        
    @property
    def keep_sh_coordinates(self):
        if self.keep_sh_levels > 1:
            full_keep_sh_coordinates = torch.cat([self._keep_sh_coordinates_dc, self._keep_sh_coordinates_rest], dim=1)
            return full_keep_sh_coordinates
        else:
            return self._keep_sh_coordinates_dc
        
    @property
    def edit_sh_coordinates(self):
        if self.edit_sh_levels > 1:
            full_edit_sh_coordinates = torch.cat([self._edit_sh_coordinates_dc, self._edit_sh_coordinates_rest], dim=1)
            return full_edit_sh_coordinates
        else:
            return self._edit_sh_coordinates_dc
        
    @property
    def scaling(self):
        full_scales = torch.cat([self._keep_scales, self._edit_scales], dim=0)
        scales = self.scale_activation(full_scales)
        return scales
    
    @property
    def quaternions(self):
        full_quaternions = torch.cat([self._keep_quaternions, self._edit_quaternions], dim=0)
        quaternions = full_quaternions
        return torch.nn.functional.normalize(quaternions, dim=-1)
    
    def area(self):
        surface_mesh = self._edit_surface_mesh
        verts = surface_mesh.verts_packed()
        faces = surface_mesh.faces_packed()
        A = verts[faces[:, 0]]
        B = verts[faces[:, 1]]
        C = verts[faces[:, 2]]
        area = triangle_area(A, B, C)
        _area = nn.Parameter(area, requires_grad=False).to(self.device)
        return _area
    
    def mean_area(self):
        _area = self.area()
        mean_area = torch.mean(_area)
        return mean_area
    
    def surface_n_gaussians(self):
        _area = self.area()
        mean_area = self.mean_area()
        n_gaussians_per_face = torch.where(_area < mean_area, torch.tensor(1, dtype=torch.int), torch.tensor(3, dtype=torch.int))
        n_gaussians_per_face = n_gaussians_per_face.view(-1, 1)  # [M, 1]
        return n_gaussians_per_face
    
    def calculate_attr_by_bary_coords(self, attr):
        '''
        attr: faces attribute
        [n_faces, None, 3, n_coords]
        '''
        surface_triangle_bary_coords_1 = torch.tensor(
            [[1/3, 1/3, 1/3]],
            dtype=torch.float32,
            device=self.nerfmodel.device,
        )[..., None]  # n_gaussians_per_face, 3, None
        surface_triangle_bary_coords_3 = torch.tensor(
            [[2/3, 1/6, 1/6],
            [1/6, 2/3, 1/6],
            [1/6, 1/6, 2/3]],
            dtype=torch.float32,
            device=self.nerfmodel.device,
        )[..., None]  # n_gaussians_per_face, 3, None
        
        attrs = []
        face_indices = []
        n_gaussians_per_face = self.surface_n_gaussians()
        n_faces = n_gaussians_per_face.shape[0]
        
        mask = (n_gaussians_per_face == 1).squeeze()
        attrs_list = []
        face_indices_list = []

        # Calculate attributes for faces with one Gaussian
        if mask.any():
            single_gaussian_attrs = (attr[mask] * surface_triangle_bary_coords_1).sum(dim=-2)
            attrs_list.append(single_gaussian_attrs.view(-1, 3))
            face_indices_list.append(torch.arange(n_faces, device=self.nerfmodel.device)[mask].unsqueeze(1).expand(-1, 1))

        # Calculate attributes for faces with three Gaussians
        if (~mask).any():
            multi_gaussian_attrs = (attr[~mask] * surface_triangle_bary_coords_3).sum(dim=-2)
            attrs_list.append(multi_gaussian_attrs.view(-1, 3))
            face_indices_list.append(torch.arange(n_faces, device=self.nerfmodel.device)[~mask].repeat_interleave(3).unsqueeze(1))

        # Concatenate results
        attrs = torch.cat(attrs_list, dim=0).reshape(-1, 3)
        face_indices = torch.cat(face_indices_list, dim=0).reshape(-1, 1)
        print("Attrs Shape:", attrs.shape)
        print("Face Indices Shape:", face_indices.shape)
        return attrs, face_indices
    
    def forward(self, **kwargs):
        pass
    
    def adapt_to_cameras(self, cameras:CamerasWrapper):
        self.focal_factor = max(cameras.p3d_cameras.K[0, 0, 0].item(),
                                cameras.p3d_cameras.K[0, 1, 1].item())
        
        self.image_height = int(cameras.height[0].item())
        self.image_width = int(cameras.width[0].item())
        
        self.min_ndc_radius = 2. / min(self.image_height, self.image_width)
        self.max_ndc_radius = 2. * 0.01
        
        self.fx = cameras.fx
        self.fy = cameras.fy
    
    def get_cameras_spatial_extent(self, nerf_cameras:CamerasWrapper=None, return_average_xyz=False):
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        
        camera_centers = nerf_cameras.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)  # Should it be replaced by the center of camera bbox, i.e. (min + max) / 2?
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        if return_average_xyz:
            return radius, avg_camera_center
        else:
            return radius
    
    def get_points_rgb(
        self,
        positions:torch.Tensor=None,
        camera_centers:torch.Tensor=None,
        directions:torch.Tensor=None,
        sh_levels:int=None,
        sh_coordinates:torch.Tensor=None,
    ):
        """Returns the RGB color of the points for the given camera pose."""
        
        if positions is None:
            positions = self.points
        if camera_centers is not None:
            render_directions = torch.nn.functional.normalize(positions - camera_centers, dim=-1)
        elif directions is not None:
            render_directions = directions
        else:
            raise ValueError("Either camera_centers or directions must be provided.")
        if sh_coordinates is None:
            raise NotImplementedError
        if sh_levels is None:
            # sh_coordinates = sh_coordinates
            raise NotImplementedError
        else:
            sh_coordinates = sh_coordinates[:, :sh_levels**2]
            
        shs_view = sh_coordinates.transpose(-1, -2).view(-1, 3, sh_levels**2)
        sh2rgb = eval_sh(sh_levels-1, shs_view, render_directions)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
        
        return colors
    
    def render_image_gaussian_rasterizer(
        self, 
        nerf_cameras:CamerasWrapper=None, 
        camera_indices:int=0,
        verbose=False,
        bg_color = None,
        sh_deg:int=None,
        sh_rotations:torch.Tensor=None,
        compute_color_in_rasterizer=False,
        compute_covariance_in_rasterizer=True,
        return_2d_radii = False,
        quaternions=None,
        use_same_scale_in_all_directions=False,
        return_opacities:bool=False,
        return_colors:bool=False,
        positions:torch.Tensor=None,
        point_colors = None,
        ):
        """Render an image using the Gaussian Splatting Rasterizer."""
        
        if nerf_cameras is None:
            nerf_cameras = self.nerf_cameras
            
        p3d_camera = nerf_cameras.p3d_cameras[camera_indices]
        fx = self.fx[camera_indices].item()
        fy = self.fy[camera_indices].item()
        fov_x = focal2fov(fx, self.image_width)
        fov_y = focal2fov(fy, self.image_height)
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(fov_x * 0.5)
        
        if bg_color is None:
            bg_color = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
            
        if positions is None:
            positions = self.points
            
        use_torch = False
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = nerf_cameras.camera_to_worlds[camera_indices]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(self.device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        world_view_transform = torch.Tensor(getWorld2View(
            R=R, t=T, tensor=use_torch)).transpose(0, 1).cuda()
        
        proj_transform = getProjectionMatrix(
            p3d_camera.znear.item(), 
            p3d_camera.zfar.item(), 
            fov_x,
            fov_y,
        ).transpose(0, 1).cuda()
        # TODO: THE TWO FOLLOWING LINES ARE IMPORTANT! IT'S NOT HERE IN 3DGS CODE! Should make a PR when I have time
        proj_transform[..., 2, 0] = - p3d_camera.K[0, 0, 2]
        proj_transform[..., 2, 1] = - p3d_camera.K[0, 1, 2]
        
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_transform.unsqueeze(0))).squeeze(0)
        
        camera_center = p3d_camera.get_camera_center()
        if verbose:
            print("p3d camera_center", camera_center)
            print("ns camera_center", nerf_cameras.camera_to_worlds[camera_indices][..., 3])
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_deg,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        if point_colors is None:
            assert (compute_color_in_rasterizer == False)
            # query rgb separately for two parts
            keep_points = self._keep_points
            edit_points = self._edit_points
            keep_sh_levels = self.keep_sh_levels
            edit_sh_levels = self.edit_sh_levels
            keep_sh_coordinates = self.keep_sh_coordinates
            edit_sh_coordinates = self.edit_sh_coordinates
            if not compute_color_in_rasterizer:
                if sh_rotations is None:
                    # keep colors
                    keep_splat_colors = self.get_points_rgb(
                        positions=keep_points,
                        camera_centers=camera_center,
                        sh_levels=keep_sh_levels,
                        sh_coordinates=keep_sh_coordinates,
                    )
                    # edit colors
                    edit_splat_colors = self.get_points_rgb(
                        positions=edit_points,
                        camera_centers=camera_center,
                        sh_levels=edit_sh_levels,
                        sh_coordinates=edit_sh_coordinates,
                    )
                    # concat two colors
                    splat_colors = torch.cat([keep_splat_colors, edit_splat_colors], dim=0)
                else:
                    raise NotImplementedError
                shs = None
            else:
                raise NotImplementedError
            
        splat_opacities = self.strengths.view(-1, 1)
        
        if quaternions is None:
            quaternions = self.quaternions
            
        if not use_same_scale_in_all_directions:
            scales = self.scaling
        else:
            scales = self.scaling.mean(dim=-1, keepdim=True).expand(-1, 3)
            scales = scales.squeeze(0)
            
        if not compute_covariance_in_rasterizer:
            cov3Dmatrix = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device=self.device)
            rotation = quaternion_to_matrix(quaternions)
            
            cov3Dmatrix[:,0,0] = scales[:,0]**2
            cov3Dmatrix[:,1,1] = scales[:,1]**2
            cov3Dmatrix[:,2,2] = scales[:,2]**2
            cov3Dmatrix = rotation @ cov3Dmatrix @ rotation.transpose(-1, -2)

            cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)
            cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
            cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
            cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
            cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
            cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
            cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
            
            quaternions = None
            scales = None
        else:
            cov3D = None
            
        screenspace_points = torch.zeros(self.n_points, 3, dtype=self._edit_points.dtype, requires_grad=True, device=self.device)
        if return_2d_radii:
            try:
                screenspace_points.retain_grad()
            except:
                print("WARNING: return_2d_radii is True, but failed to retain grad of screenspace_points!")
                pass
        means2D = screenspace_points
        
        if verbose:
            print("points", positions.shape)
            if not compute_color_in_rasterizer:
                print("splat_colors", splat_colors.shape)
            print("splat_opacities", splat_opacities.shape)
            print("quaternions", quaternions.shape)
            print("scales", scales.shape)
            print("screenspace_points", screenspace_points.shape)
            
        rendered_image, radii = rasterizer(
            means3D = positions,
            means2D = means2D,
            shs = shs,
            colors_precomp = splat_colors,
            opacities = splat_opacities,
            scales = scales,
            rotations = quaternions,
            cov3D_precomp = cov3D
        )
        
        if not(return_2d_radii or return_opacities or return_colors):
            return rendered_image.transpose(0, 1).transpose(1, 2)
        
        else:
            outputs = {
                "image": rendered_image.transpose(0, 1).transpose(1, 2),
                "radii": radii,
                "viewspace_points": screenspace_points,
            }
            if return_opacities:
                outputs["opacities"] = splat_opacities
            if return_colors:
                outputs["colors"] = splat_colors
        
            return outputs
        
    def save_model(self, path, **kwargs):
        checkpoint = {}
        checkpoint['state_dict'] = self.state_dict()
        for k, v in kwargs.items():
            print(k)
            checkpoint[k] = v
        torch.save(checkpoint, path)  
        
        
def load_inpainted_model(refined_tetgs_path, nerfmodel:GaussianSplattingWrapper):
    checkpoint = torch.load(refined_tetgs_path, map_location=nerfmodel.device)
    n_all_faces = checkpoint['state_dict']['_surface_mesh_faces'].shape[0]
    n_all_vertices = checkpoint['state_dict']['_verts_points'].shape[0]
    n_edit_faces = checkpoint['state_dict']['_edit_mesh_faces'].shape[0]
    n_edit_vertices = checkpoint['state_dict']['_edit_mesh_vertices'].shape[0]
    keep_faces_num = n_all_faces - n_edit_faces
    keep_vertices_num = n_all_vertices - n_edit_vertices
    
    n_edit_gaussians = checkpoint['state_dict']['_edit_scales'].shape[0]
    n_keep_gaussians = checkpoint['state_dict']['_keep_scales'].shape[0]
    
    print("Loading refined model...")
    print(f'{n_all_faces} all faces detected.')
    print(f'{n_edit_faces} edit faces detected.')
    print(f'{n_edit_gaussians} edit gaussians detected.')
    print(f'{n_keep_gaussians} keep gaussians detected.')
    
    if checkpoint['state_dict']['_verts_points'].shape[1] == 3:
        _points = checkpoint['state_dict']['_verts_points']
    else:
        _points = checkpoint['state_dict']['_verts_points'].repeat(1, 3)
    
    with torch.no_grad():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(_points.cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(checkpoint['state_dict']['_surface_mesh_faces'].cpu().numpy())
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(_points).cpu().numpy())
    
    keep_xyz = checkpoint['state_dict']['_keep_points'].cpu().numpy()
    keep_opacities = checkpoint['state_dict']['all_keep_densities'].cpu().numpy()
    keep_scales = checkpoint['state_dict']['_keep_scales'].cpu().numpy()
    keep_rots = checkpoint['state_dict']['_keep_quaternions'].cpu().numpy()
    keep_sh_coordinates_dc = checkpoint['state_dict']['_keep_sh_coordinates_dc'].cpu().numpy()
    keep_face_indices = checkpoint['state_dict']['_keep_face_indices'].cpu().numpy()
    sh_level = 4   # checkpoint['state_dict']['keep_sh_levels'].cpu().numpy()
    if sh_level > 1:
        keep_sh_coordinates_rest = checkpoint['state_dict']['_keep_sh_coordinates_rest'].cpu().numpy()
    
    keep_gaussians = {
        "keep_xyz": torch.tensor(keep_xyz, dtype=torch.float).to(nerfmodel.device),
        "keep_opacities": torch.tensor(keep_opacities, dtype=torch.float).to(nerfmodel.device),
        "keep_scales": torch.tensor(keep_scales, dtype=torch.float).to(nerfmodel.device),
        "keep_rots": torch.tensor(keep_rots, dtype=torch.float).to(nerfmodel.device),
        "keep_sh_coordinates_dc": torch.tensor(keep_sh_coordinates_dc, dtype=torch.float).to(nerfmodel.device),
        "keep_face_indices": torch.tensor(keep_face_indices, dtype=torch.float).to(nerfmodel.device),
        "sh_level": sh_level
    }
    if sh_level > 1:
        keep_gaussians.update(
            {
                "keep_sh_coordinates_rest": torch.tensor(keep_sh_coordinates_rest, dtype=torch.float).to(nerfmodel.device),
            }
        )
    
    refined_tetgs = EditTetGS(
        nerfmodel=nerfmodel,
        points=_points,
        colors=torch.ones((_points.shape[0], 3), dtype=torch.float).to(nerfmodel.device),
        keep_gaussians=keep_gaussians,
        keep_faces_num=keep_faces_num,
        keep_vertices_num=keep_vertices_num,
        initialize=False,
        sh_levels=1,
        surface_mesh_to_bind=o3d_mesh
    )
    refined_tetgs.load_state_dict(checkpoint['state_dict'], strict=False)
    
    return refined_tetgs


def convert_edited_tetgs_into_masked_gaussians(edited_tetgs: EditTetGS):
    '''
    input:
        edited_2dgs_tetgs
    output:
        分开取出ckpt里keep_gaussians和edit_gaussians的所有attrs
    '''
    
    edit_sh_level = edited_tetgs.edit_sh_levels
    keep_sh_level = edited_tetgs.keep_sh_levels
    
    with torch.no_grad():
        # keep part
        keep_xyz = edited_tetgs._keep_points.cpu().numpy()
        keep_opacities = edited_tetgs.all_keep_densities.cpu().numpy()
        keep_scales = edited_tetgs._keep_scales.cpu().numpy()
        keep_rots = edited_tetgs._keep_quaternions.cpu().numpy()
        keep_sh_coordinates_dc = edited_tetgs._keep_sh_coordinates_dc.cpu().numpy()
        keep_face_indices = edited_tetgs._keep_face_indices.cpu().numpy()
        if keep_sh_level > 1:
            keep_sh_coordinates_rest = edited_tetgs._keep_sh_coordinates_rest.cpu().numpy()
            
        # edit part
        edit_xyz = edited_tetgs._edit_points.cpu().numpy()
        edit_opacities = edited_tetgs.all_edit_densities.cpu().numpy()
        edit_scales = edited_tetgs._edit_scales.cpu().numpy()
        edit_rots = edited_tetgs._edit_quaternions.cpu().numpy()
        edit_sh_coordinates_dc = edited_tetgs._edit_sh_coordinates_dc.cpu().numpy()
        edit_face_indices = edited_tetgs._edit_face_indices.cpu().numpy()
        if edit_sh_level > 1:
            edit_sh_coordinates_rest = edited_tetgs._edit_sh_coordinates_rest.cpu().numpy()
            
    edited_all_gaussians = {}
    
    # keep part
    edited_all_gaussians.update(
        {
            "keep_xyz": torch.tensor(keep_xyz, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "keep_opacities": torch.tensor(keep_opacities, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "keep_scales": torch.tensor(keep_scales, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "keep_rots": torch.tensor(keep_rots, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "keep_sh_coordinates_dc": torch.tensor(keep_sh_coordinates_dc, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "keep_face_indices": torch.tensor(keep_face_indices, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "keep_sh_level": keep_sh_level,
        }
    )
    if keep_sh_level > 1:
        edited_all_gaussians.update(
            {
                "keep_sh_coordinates_rest": torch.tensor(keep_sh_coordinates_rest, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            }
        )
        
    # edit part
    edited_all_gaussians.update(
        {
            "edit_xyz": torch.tensor(edit_xyz, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "edit_opacities": torch.tensor(edit_opacities, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "edit_scales": torch.tensor(edit_scales, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "edit_rots": torch.tensor(edit_rots, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "edit_sh_coordinates_dc": torch.tensor(edit_sh_coordinates_dc, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "edit_face_indices": torch.tensor(edit_face_indices, dtype=torch.float).to(edited_tetgs.nerfmodel.device),
            "edit_sh_level": edit_sh_level,
        }
    )
    if edit_sh_level > 1:
        edited_all_gaussians.update(
            {
                "edit_sh_coordinates_rest": torch.tensor(edit_sh_coordinates_rest, dtype=torch.float).to(edited_tetgs.nerfmodel.device)
            }
        )
        
    return edited_all_gaussians