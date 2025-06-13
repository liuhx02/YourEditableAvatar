import torch.nn as nn
import open3d as o3d
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.ops import knn_points
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.spherical_harmonics import (
    eval_sh, RGB2SH, SH2RGB,
)
from utils.graphics_utils import *
from utils.general_utils import inverse_sigmoid
from tetgs_scene.gs_model import GaussianSplattingWrapper
from tetgs_scene.cameras import CamerasWrapper

scale_activation = torch.exp
scale_inverse_activation = torch.log
use_old_method = False


def _initialize_radiuses_gauss_rasterizer(tetgs):
    # Initialize learnable radiuses
    tetgs.image_height = int(tetgs.nerfmodel.training_cameras.height[0].item())
    tetgs.image_width = int(tetgs.nerfmodel.training_cameras.width[0].item())
    
    all_camera_centers = tetgs.nerfmodel.training_cameras.camera_to_worlds[..., 3]
    all_camera_dists = torch.cdist(tetgs.points, all_camera_centers)[None]
    d_charac = all_camera_dists.mean(-1, keepdim=True)
    
    ndc_factor = 1.
    tetgs.min_ndc_radius = ndc_factor * 2. / min(tetgs.image_height, tetgs.image_width)
    tetgs.max_ndc_radius = ndc_factor * 2. * 0.05  # 2. * 0.01
    tetgs.min_radius = tetgs.min_ndc_radius / tetgs.focal_factor * d_charac
    tetgs.max_radius = tetgs.max_ndc_radius / tetgs.focal_factor * d_charac
    
    knn = knn_points(tetgs.points[None], tetgs.points[None], K=4)
    use_sqrt = True
    use_mean = False
    initial_radius_normalization = 1.  # 1., 0.1
    if use_sqrt:
        knn_dists = torch.sqrt(knn.dists[..., 1:])
    else:
        knn_dists = knn.dists[..., 1:]
    if use_mean:
        print("Use mean to initialize scales.")
        radiuses = knn_dists.mean(-1, keepdim=True).clamp_min(0.0000001) * initial_radius_normalization
    else:
        print("Use min to initialize scales.")
        radiuses = knn_dists.min(-1, keepdim=True)[0].clamp_min(0.0000001) * initial_radius_normalization
    
    res = inverse_radius_fn(radiuses=radiuses)
    tetgs.radius_dim = res.shape[-1]
    
    return res


class TetGS(nn.Module):
    def __init__(
        self, 
        nerfmodel: GaussianSplattingWrapper,
        points: torch.Tensor,
        colors: torch.Tensor,
        initialize:bool=True,
        sh_levels:int=4,
        learnable_positions:bool=True,
        keep_track_of_knn:bool=False,
        knn_to_track:int=16,
        learn_color_only=False,
        freeze_gaussians=False,
        surface_mesh_to_bind=None,  # Open3D mesh
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=True,
        learn_surface_mesh_opacity=True,
        learn_surface_mesh_scales=True,
        face_to_global_tet_idx=None,
        *args, **kwargs) -> None:

        super(TetGS, self).__init__()
        
        self.nerfmodel = nerfmodel
        self.freeze_gaussians = freeze_gaussians
        
        self.learn_positions = ((not learn_color_only) and learnable_positions) and (not freeze_gaussians)
        self.learn_opacities = (not learn_color_only) and (not freeze_gaussians)
        self.learn_scales = (not learn_color_only) and (not freeze_gaussians)
        self.learn_quaternions = (not learn_color_only) and (not freeze_gaussians)
        self.learnable_positions = learnable_positions
        
        if surface_mesh_to_bind is not None:
            self.learn_surface_mesh_positions = learn_surface_mesh_positions
            self.binded_to_surface_mesh = True
            self.bind_3dgs = True 
            self.learn_surface_mesh_opacity = learn_surface_mesh_opacity
            self.learn_surface_mesh_scales = learn_surface_mesh_scales
            
            self.learn_positions = self.learn_surface_mesh_positions
            self.learn_scales = self.learn_surface_mesh_scales
            self.learn_quaternions = self.learn_surface_mesh_scales
            self.learn_opacities = self.learn_surface_mesh_opacity
            
            # mesh
            self._surface_mesh_faces = torch.nn.Parameter(
                torch.tensor(np.array(surface_mesh_to_bind.triangles)).to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            if surface_mesh_thickness is None:
                surface_mesh_thickness = nerfmodel.training_cameras.get_spatial_extent() / 1_000_000
            self.surface_mesh_thickness = torch.nn.Parameter(
                torch.tensor(surface_mesh_thickness).to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            verts_points = torch.tensor(np.array(surface_mesh_to_bind.vertices)).float().to(nerfmodel.device)
            self._verts_points = torch.nn.Parameter(
                verts_points.to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            
            # face_to_global_tet_idx
            if face_to_global_tet_idx is not None:
                self.face_to_global_tet_idx = torch.nn.Parameter(
                    face_to_global_tet_idx, requires_grad=False).to(nerfmodel.device)
            
            # color
            if surface_mesh_to_bind.vertex_colors is None or len(surface_mesh_to_bind.vertex_colors) == 0:
                # Randomly initialize colors if vertex_colors is empty
                num_vertices = len(surface_mesh_to_bind.vertices)
                gray_colors = np.ones((num_vertices, 3)) * 0.5
                surface_mesh_to_bind.vertex_colors = o3d.utility.Vector3dVector(gray_colors)
            self._vertex_colors = torch.tensor(np.array(surface_mesh_to_bind.vertex_colors)).float().to(nerfmodel.device)
            self._surface_mesh = Meshes(
                verts=[verts_points.to(self.device)],   
                faces=[self._surface_mesh_faces.to(self.device)],
                textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
                )
            faces_colors = self._vertex_colors[self._surface_mesh_faces]  # n_faces, 3, n_coords
            colors, _ = self.calculate_attr_by_bary_coords(faces_colors[:, None])
            
            update_normal = True
            self.update_normal = update_normal
            
            if not update_normal:
                self._points_mesh = verts_points
                # First gather vertices of all triangles
                faces_verts = self._points_mesh[self._surface_mesh_faces]
                points, face_indices = self.calculate_attr_by_bary_coords(faces_verts[:, None])
                self._points = nn.Parameter(points, requires_grad=self.learn_positions).to(nerfmodel.device)
                self._face_indices = face_indices

                n_points = points.shape[0]
                self._n_points = n_points
            else:
                self._points_mesh = verts_points
                faces_verts = self._points_mesh[self._surface_mesh_faces]
                points, face_indices = self.calculate_attr_by_bary_coords(faces_verts[:, None])
                self._face_indices = face_indices

                n_points = points.shape[0]
                self._n_points = n_points
                self.ori_points = torch.nn.Parameter(
                    points.to(nerfmodel.device),
                    requires_grad=False).to(nerfmodel.device)
                
                # normal
                verts_normals = torch.nn.functional.normalize(self._surface_mesh.verts_normals_list()[0], dim=-1).view(-1, 1, 3)
                verts_normals = verts_normals.reshape(-1, 3)
                self._mesh_normal = verts_normals
                faces_normals = self._mesh_normal[self._surface_mesh_faces]
                normals, _ = self.calculate_attr_by_bary_coords(faces_normals[:, None])
                self.normals = torch.nn.Parameter(
                    normals.to(nerfmodel.device),
                    requires_grad=False).to(nerfmodel.device)
                
                # delta
                deltas = torch.zeros(self._n_points, 1).to(nerfmodel.device)
                self._points = nn.Parameter(deltas, requires_grad=self.learn_positions).to(nerfmodel.device)        
        else:
            raise NotImplementedError("TetGS must be binded to a mesh")
        
        # KNN information for training regularization
        self.keep_track_of_knn = keep_track_of_knn
        if keep_track_of_knn:
            self.knn_to_track = knn_to_track
            knns = knn_points(points[None], points[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]
        
        # Render parameters
        self.image_height = int(nerfmodel.training_cameras.height[0].item())
        self.image_width = int(nerfmodel.training_cameras.width[0].item())
        self.focal_factor = max(nerfmodel.training_cameras.p3d_cameras.K[0, 0, 0].item(),
                                nerfmodel.training_cameras.p3d_cameras.K[0, 1, 1].item())
        
        self.fx = nerfmodel.training_cameras.fx[0].item()
        self.fy = nerfmodel.training_cameras.fy[0].item()
        self.fov_x = focal2fov(self.fx, self.image_width)
        self.fov_y = focal2fov(self.fy, self.image_height)
        self.tanfovx = math.tan(self.fov_x * 0.5)
        self.tanfovy = math.tan(self.fov_y * 0.5)
        
        if self.binded_to_surface_mesh and (not learn_surface_mesh_opacity):
            all_densities = inverse_sigmoid(0.9999 * torch.ones((n_points, 1), dtype=torch.float, device=points.device))
            self.learn_opacities = False
        else:
            all_densities = inverse_sigmoid(0.1 * torch.ones((n_points, 1), dtype=torch.float, device=points.device))
        self.all_densities = nn.Parameter(all_densities, 
                                     requires_grad=self.learn_opacities).to(nerfmodel.device)
        self.return_one_densities = False
        
        self.min_ndc_radius = 2. / min(self.image_height, self.image_width)
        self.max_ndc_radius = 2. * 0.01  # 2. * 0.01
        self.min_radius = None # self.min_ndc_radius / self.focal_factor * 0.005  # 0.005
        self.max_radius = None # self.max_ndc_radius / self.focal_factor * 2.  # 2.
        
        self.radius_dim = 7
        
        # Initialize learnable radiuses
        self.scale_activation = scale_activation
        self.scale_inverse_activation = scale_inverse_activation
        if initialize:
            radiuses = _initialize_radiuses_gauss_rasterizer(self,)
            print("Initialized radiuses for 3D Gauss Rasterizer")
        else:
            radiuses = torch.rand(1, n_points, self.radius_dim, device=nerfmodel.device)
            self.min_radius = self.min_ndc_radius / self.focal_factor * 0.005 
            self.max_radius = self.max_ndc_radius / self.focal_factor * 2. 
        # 3D Gaussian parameters
        self._scales = nn.Parameter(
            radiuses[0, ..., 4:],
            requires_grad=self.learn_scales).to(nerfmodel.device)
        self._quaternions = nn.Parameter(
            radiuses[0, ..., :4],
            requires_grad=self.learn_quaternions).to(nerfmodel.device)
        
        # Initialize color features
        self.sh_levels = sh_levels
        sh_coordinates_dc = RGB2SH(colors).unsqueeze(dim=1)
        self._sh_coordinates_dc = nn.Parameter(
            sh_coordinates_dc.to(self.nerfmodel.device),
            requires_grad=True and (not freeze_gaussians)
        ).to(self.nerfmodel.device)
        if sh_levels > 1:
            self._sh_coordinates_rest = nn.Parameter(
                torch.zeros(n_points, sh_levels**2 - 1, 3).to(self.nerfmodel.device),
                requires_grad=True and (not freeze_gaussians)
            ).to(self.nerfmodel.device)
    
    @property
    def device(self):
        return self.nerfmodel.device
    
    @property
    def n_points(self):
        return self._n_points
    
    @property
    def points(self):
        if not self.update_normal:
            return self._points
        else:
            new_pos = self.ori_points + self.normals * self._points
            return new_pos
    
    @property
    def strengths(self):
        if self.return_one_densities:
            return torch.ones_like(self.all_densities.view(-1, 1))
        else:
            return torch.sigmoid(self.all_densities.view(-1, 1))
        
    @property
    def sh_coordinates(self):
        if self.sh_levels > 1:
            return torch.cat([self._sh_coordinates_dc, self._sh_coordinates_rest], dim=1)
        else:
            return self._sh_coordinates_dc
    
    @property
    def radiuses(self):
        return torch.cat([self._quaternions, self._scales], dim=-1)[None]
    
    @property
    def scaling(self):
        scales = self.scale_activation(self._scales)
        return scales
    
    @property
    def quaternions(self):
        quaternions = self._quaternions
        return torch.nn.functional.normalize(quaternions, dim=-1)

        
    @property
    def surface_mesh(self):
        # Create a Meshes object
        surface_mesh = Meshes(
            verts=[self._points_mesh.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            )
        return surface_mesh
    
    @property
    def radii(self):
        # scaling reg
        verts = self.surface_mesh.verts_packed()
        faces = self.surface_mesh.faces_packed()
        A = verts[faces[:, 0]]
        B = verts[faces[:, 1]]
        C = verts[faces[:, 2]]
        radii = circumcircle_radius(A, B, C)
        gaussian_radii = radii[self._face_indices].reshape(-1)
        _radii = nn.Parameter(gaussian_radii, requires_grad=False).to(self.device)
        return _radii
    
    def area(self):
        surface_mesh = self._surface_mesh
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
        
        self.fx = cameras.fx[0].item()
        self.fy = cameras.fy[0].item()
        self.fov_x = focal2fov(self.fx, self.image_width)
        self.fov_y = focal2fov(self.fy, self.image_height)
        self.tanfovx = math.tan(self.fov_x * 0.5)
        self.tanfovy = math.tan(self.fov_y * 0.5)
        
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
        if positions is None:
            positions = self.points

        if camera_centers is not None:
            render_directions = torch.nn.functional.normalize(positions - camera_centers, dim=-1)
        elif directions is not None:
            render_directions = directions
        else:
            raise ValueError("Either camera_centers or directions must be provided.")

        if sh_coordinates is None:
            sh_coordinates = self.sh_coordinates 
        if sh_levels is None:
            sh_coordinates = sh_coordinates
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
        return_alphas:bool=False,
        return_depths:bool=False,
        positions:torch.Tensor=None,
        point_colors = None
        ):
        """Render an image using the Gaussian Splatting Rasterizer."""

        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras

        p3d_camera = nerf_cameras.p3d_cameras[camera_indices]

        if bg_color is None:
            bg_color = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
            
        if positions is None:
            positions = self.points

        use_torch = True
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = nerf_cameras.camera_to_worlds[camera_indices]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(self.device)], dim=0) #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        c2w = c2w.squeeze()

        # get the world-to-camera transform and set R, T
        w2c = torch.inverse(c2w) 
        R = w2c[:3, :3].T 
        T = w2c[:3, 3]
        world_view_transform = torch.Tensor(getWorld2View(
            R=R, t=T, tensor=use_torch)).transpose(0, 1)
        
        proj_transform = getProjectionMatrix(
            p3d_camera.znear.item(), 
            p3d_camera.zfar.item(), 
            self.fov_x, 
            self.fov_y).transpose(0, 1).cuda()
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
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
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

        # TODO: Change color computation to match 3DGS paper (remove sigmoid)
        if point_colors is None:
            if not compute_color_in_rasterizer:
                if sh_rotations is None:
                    splat_colors = self.get_points_rgb(
                        positions=positions, 
                        camera_centers=camera_center,
                        sh_levels=sh_deg+1,)
                else:
                    splat_colors = self.get_points_rgb(
                        positions=positions, 
                        camera_centers=None,
                        directions=(torch.nn.functional.normalize(positions - camera_center, dim=-1).unsqueeze(1) @ sh_rotations)[..., 0, :],
                        sh_levels=sh_deg+1,)
                shs = None
            else:
                shs = self.sh_coordinates
                splat_colors = None
        else:
            splat_colors = point_colors
            shs = None
            
        splat_opacities = self.strengths.view(-1, 1)
        
        if quaternions is None:
            quaternions = self.quaternions
        
        if not use_same_scale_in_all_directions:
            scales = self.scaling
        else:
            scales = self.scaling.mean(dim=-1, keepdim=True).expand(-1, 3)
            scales = scales.squeeze(0)
        
        if verbose:
            print("Scales:", scales.shape, scales.min(), scales.max())

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
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(self._points, dtype=self._points.dtype, requires_grad=True, device=self.device) + 0
        screenspace_points = torch.zeros(self.n_points, 3, dtype=self._points.dtype, requires_grad=True, device=self.device)
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
            if not compute_covariance_in_rasterizer:
                print("cov3D", cov3D.shape)
                print(cov3D[0])
            else:
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
        
        if not(return_2d_radii or return_opacities or return_colors or return_alphas or return_depths):
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
            if return_alphas:
                outputs["alphas"] = None
            if return_depths:
                outputs["depths"] = None
        
            return outputs

    def save_model(self, path, **kwargs):
        checkpoint = {}
        checkpoint['state_dict'] = self.state_dict()
        for k, v in kwargs.items():
            checkpoint[k] = v
        torch.save(checkpoint, path)  


def load_init_model(refined_tetgs_path, nerfmodel:GaussianSplattingWrapper):
    checkpoint = torch.load(refined_tetgs_path, map_location=nerfmodel.device)
    n_faces = checkpoint['state_dict']['_surface_mesh_faces'].shape[0]
    n_gaussians = checkpoint['state_dict']['_scales'].shape[0]
    face_to_global_tet_idx = checkpoint['state_dict']['face_to_global_tet_idx']

    print("Loading refined model...")
    print(f'{n_faces} faces detected.')
    print(f'{n_gaussians} gaussians detected.')
    if checkpoint['state_dict']['_verts_points'].shape[1] == 3:
        _points = checkpoint['state_dict']['_verts_points']
    else:
        _points = checkpoint['state_dict']['_verts_points'].repeat(1, 3)

    with torch.no_grad():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(_points.cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(checkpoint['state_dict']['_surface_mesh_faces'].cpu().numpy())
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(_points).cpu().numpy())
        
    refined_tetgs = TetGS(
        nerfmodel=nerfmodel,
        points=_points,
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=4,
        keep_track_of_knn=False,
        knn_to_track=0,
        surface_mesh_to_bind=o3d_mesh,
        face_to_global_tet_idx=face_to_global_tet_idx
    )
    refined_tetgs.load_state_dict(checkpoint['state_dict'])
    return refined_tetgs


# TODO: fix this
def convert_refined_tetgs_into_masked_gaussians_old(refined_tetgs: TetGS, keep_faces_num):
    '''
    input:
        refined_tetgs: 重建训练完的tetgs ckpt
        keep_faces_num: keep_mesh的面片数量
        
        在重建用mesh和编辑后mesh中, keep部分的顶点和面片都排在edit前面
        verts: [keep_verts, edit_verts]
        faces: [keep_faces, edit_faces]
        
        只需取出面片编号在[0:keep_faces_num]的高斯球即可
    '''
    
    keep_gaussians = {}
    sh_level = refined_tetgs.sh_levels
    
    with torch.no_grad():
        xyz = refined_tetgs.points.cpu().numpy()
        opacities = refined_tetgs.all_densities.cpu().numpy()
        scales = scale_inverse_activation(refined_tetgs.scaling).cpu().numpy()
        rots = refined_tetgs.quaternions.cpu().numpy()
        sh_coordinates_dc = refined_tetgs._sh_coordinates_dc.cpu().numpy()
        if sh_level > 1:
            sh_coordinates_rest = refined_tetgs._sh_coordinates_rest.cpu().numpy()
        # face_indices
        face_indices = refined_tetgs._face_indices.cpu().numpy()
        
    keep_gaussian_indices = np.where((face_indices < keep_faces_num))[0]
    # extract keep_attrs
    keep_xyz = xyz[keep_gaussian_indices]
    keep_opacities = opacities[keep_gaussian_indices]
    keep_scales = scales[keep_gaussian_indices]
    keep_rots = rots[keep_gaussian_indices]
    keep_sh_coordinates_dc = sh_coordinates_dc[keep_gaussian_indices]
    keep_face_indices = face_indices[keep_gaussian_indices]
    if sh_level > 1:
        keep_sh_coordinates_rest = sh_coordinates_rest[keep_gaussian_indices]
        
    keep_gaussians.update(
        {
            "keep_xyz": torch.tensor(keep_xyz, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_opacities": torch.tensor(keep_opacities, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_scales": torch.tensor(keep_scales, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_rots": torch.tensor(keep_rots, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_sh_coordinates_dc": torch.tensor(keep_sh_coordinates_dc, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_face_indices": torch.tensor(keep_face_indices, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "sh_level": sh_level
        }
    )
    if sh_level > 1:
        keep_gaussians.update(
            {
                "keep_sh_coordinates_rest": torch.tensor(keep_sh_coordinates_rest, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            }
        )
    
    return keep_gaussians

# inherit keep gaussians from reconstructed ones
def convert_refined_tetgs_into_masked_gaussians(refined_tetgs: TetGS, edit_face_to_global_tet_idx):
    keep_gaussians = {}
    sh_level = refined_tetgs.sh_levels
    with torch.no_grad():
        xyz = refined_tetgs.points.cpu().numpy()
        opacities = refined_tetgs.all_densities.cpu().numpy()
        scales = scale_inverse_activation(refined_tetgs.scaling).cpu().numpy()
        rots = refined_tetgs.quaternions.cpu().numpy()
        sh_coordinates_dc = refined_tetgs._sh_coordinates_dc.cpu().numpy()
        if sh_level > 1:
            sh_coordinates_rest = refined_tetgs._sh_coordinates_rest.cpu().numpy()
        face_indices = refined_tetgs._face_indices.cpu().numpy()
        face_to_global_tet_idx = refined_tetgs.face_to_global_tet_idx.cpu().numpy()
    
    if isinstance(edit_face_to_global_tet_idx, torch.Tensor):
        edit_face_to_global_tet_idx = edit_face_to_global_tet_idx.detach().cpu().numpy()
    inherit_tet_idx = np.intersect1d(face_to_global_tet_idx, edit_face_to_global_tet_idx)
    
    face_mask = np.isin(face_to_global_tet_idx, edit_face_to_global_tet_idx)
    inherit_face_indices = np.where(face_mask)[0]
    gaussians_mask = np.isin(face_indices, inherit_face_indices)
    inherit_gaussian_indices = np.where(gaussians_mask)[0]
    
    # extract keep_attrs
    keep_xyz = xyz[inherit_gaussian_indices]
    keep_opacities = opacities[inherit_gaussian_indices]
    keep_scales = scales[inherit_gaussian_indices]
    keep_rots = rots[inherit_gaussian_indices]
    keep_sh_coordinates_dc = sh_coordinates_dc[inherit_gaussian_indices]
    keep_face_indices = face_indices[inherit_gaussian_indices]
    if sh_level > 1:
        keep_sh_coordinates_rest = sh_coordinates_rest[inherit_gaussian_indices]
        
    keep_gaussians.update(
        {
            "keep_xyz": torch.tensor(keep_xyz, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_opacities": torch.tensor(keep_opacities, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_scales": torch.tensor(keep_scales, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_rots": torch.tensor(keep_rots, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_sh_coordinates_dc": torch.tensor(keep_sh_coordinates_dc, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "keep_face_indices": torch.tensor(keep_face_indices, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
            "sh_level": sh_level
        }
    )
    if sh_level > 1:
        keep_gaussians.update({
            "keep_sh_coordinates_rest": torch.tensor(keep_sh_coordinates_rest, dtype=torch.float).to(refined_tetgs.nerfmodel.device),
        })
    return keep_gaussians