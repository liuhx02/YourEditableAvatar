import numpy as np
import torch
import torch.nn as nn
import tetgs_spatial
from tetgs_spatial.models.mesh import Mesh
from tetgs_spatial.utils.typing import *


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)
    @property
    def grid_vertices(self) -> Float[Tensor, "N 3"]:
        raise NotImplementedError

class MarchingTetrahedraHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, tets_path: str):
        super().__init__()
        self.resolution = resolution
        self.tets_path = tets_path
        self.triangle_table: Float[Tensor, "..."]
        self.register_buffer(
            "triangle_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_table",
            torch.as_tensor(
                [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long
            ),
            persistent=False,
        )
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer(
            "base_tet_edges",
            torch.as_tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long),
            persistent=False,
        )

        tets = np.load(self.tets_path)
        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            torch.from_numpy(tets["vertices"]).float(),
            persistent=False,
        )
        self.indices: Integer[Tensor, "..."]
        self.register_buffer(
            "indices", torch.from_numpy(tets["indices"]).long(), persistent=False
        )
        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0])
            / (self.resolution)  # half tet size is approximately 1 / self.resolution
            * torch.tanh(grid_vertex_offsets)
        )

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=torch.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)
    
    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device
                ) * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=pos_nx3.device
            )
            idx_map = mapping[idx_map]  # map edges to verts
            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        # number of triangles from each valid tet
        num_triangles = self.num_triangles_table[tetindex]  # [valid_tets.shape[0], 1]

        # Generate triangle indices: map faces to tets
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )
        
        valid_tet_indices = torch.where(valid_tets)[0]
        tet_with_one = valid_tet_indices[num_triangles == 1]
        tet_with_two = valid_tet_indices[num_triangles == 2]
        face_to_tet_idx = torch.cat([
            tet_with_one,                   
            tet_with_two.repeat_interleave(2) 
        ]) # [faces.shape[0]]
        
        outputs = {
            "verts": verts,   # 被插值出来的mesh顶点坐标
            "faces": faces,   # 应该和tet的顺序也是对应的, 只不过有的tet有一个face, 有的有两个
            "face_to_tet_idx": face_to_tet_idx, # 这样我可以直接找到faces和valid_tets的对应关系诶
            "valid_tets": valid_tets,  # sdf有效的四面体序号
            "interp_v": interp_v,   # 用于插值mesh顶点的四面体顶点索引, 顺序和verts是对应的
        }
        return outputs

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(deformation)
        else:
            grid_vertices = self.grid_vertices
        outputs = self._forward(grid_vertices, level, self.indices)
        v_pos = outputs["verts"]
        t_pos_idx = outputs["faces"]
        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
        )
        return mesh
    
    def mark_part_tets(
        self,
        deformation,
        level,
        edit_mask,
    ):
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(deformation)
        else:
            grid_vertices = self.grid_vertices
            
        outputs = self._forward(grid_vertices, level, self.indices)
        face_to_tet_idx = outputs["face_to_tet_idx"]
        keep_faces_idx = torch.where(edit_mask == 0)[0]
        keep_tet_idx = face_to_tet_idx[keep_faces_idx]
        keep_tet_idx = torch.unique(keep_tet_idx)
        keep_tet_verts_idx = self.indices[keep_tet_idx, :].reshape(-1)
        keep_verts_indices, keep_idx_map = torch.unique(keep_tet_verts_idx, dim=0, return_inverse=True)
        keep_pos = grid_vertices[keep_verts_indices]
        keep_sdf = level[keep_verts_indices]
        keep_tets = keep_idx_map.reshape(-1, 4)
        
        unmapped_mask = torch.ones(self.indices.shape[0], dtype=torch.bool)
        unmapped_mask[keep_tet_idx] = False
        update_tet_idx = torch.nonzero(unmapped_mask, as_tuple=True)[0]
        update_tet_verts_idx = self.indices[update_tet_idx].reshape(-1)
        update_verts_indices, update_idx_map = torch.unique(update_tet_verts_idx, dim=0, return_inverse=True)
        new_pos = grid_vertices[update_verts_indices]
        new_sdf = level[update_verts_indices]
        new_tets = update_idx_map.reshape(-1, 4)
        
        outputs = {
             # frozen tets
            "keep_verts_indices": keep_verts_indices,  # 不更新的四面体对应的顶点索引
            "keep_pos": keep_pos,  # frozen tet vertices
            "keep_sdf": keep_sdf,
            "keep_tets": keep_tets,
            "keep_tet_idx": keep_tet_idx,   # torch.unique(keep_tet_idx),
            # edit tets
            "unique_grid_vtx": update_verts_indices,  # 更新的四面体对应的顶点索引 
            "new_pos": new_pos,  # edit tet vertices (overlapped)
            "new_sdf": new_sdf,
            "new_tets": new_tets,
        }
        # 记录new_pos中和keep_pos重叠的部分, 从new_pos中刨除这部分
        keep_pos_set = set(map(tuple, keep_pos.cpu().numpy()))
        mask = np.array([tuple(vertex) in keep_pos_set for vertex in new_pos.cpu().numpy()])
        mask_keep_part_in_new_pos = torch.from_numpy(mask).int().reshape(-1, 1).to(keep_pos.device)
        outputs.update(
            {
                "mask_keep_part_in_new_pos": mask_keep_part_in_new_pos
            }
        )
        return outputs
    
    # compact tet grids for more effcient subdivision
    def compact_tets(self, pos_nx3, sdf_n, tet_fx4, mask_keep_part_in_new_pos=None):
        with torch.no_grad():
            sdf_fx4 = sdf_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            sdf_mean = torch.mean(sdf_fx4, -1)
            sdf_mean = torch.abs(sdf_mean)
            valid_tets = (sdf_mean <= 0.02) 
            
            valid_grid_vtx = tet_fx4[valid_tets].reshape(-1)
            unique_grid_vtx, idx_map = torch.unique(valid_grid_vtx, dim=0, return_inverse=True)
            # new compacted pos
            new_pos = pos_nx3[unique_grid_vtx]  # tet verts
            # new compacted mask
            if mask_keep_part_in_new_pos is not None:
                new_mask = mask_keep_part_in_new_pos[unique_grid_vtx]
            else:
                new_mask = None
            # new compacted sdf
            new_sdf = sdf_n[unique_grid_vtx]   # sdf of tet verts
            new_tets = idx_map.reshape(-1, 4)  # 4 vert's indices of each tet
            new_tet_idx_to_old = torch.where(valid_tets)[0] # maps new_tets idx to tet_fx4 idx
            return new_pos, new_sdf, new_tets, new_mask, new_tet_idx_to_old
    
    # 8x subdivision around surface     
    def batch_subdivide_volume(self, tet_pos_bxnx3, tet_bxfx4, mask_keep_part_in_new_pos=None):
        device = tet_pos_bxnx3.device
        # get new tets
        tet_fx4 = tet_bxfx4[0]
        # get all new edges on new tets4
        all_edges = tet_fx4[:, self.base_tet_edges].reshape(-1, 2)
        all_edges = self.sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        idx_map = idx_map + tet_pos_bxnx3.shape[1]
        # get new verts' position
        all_values = tet_pos_bxnx3
        # midpoints of all new edges
        mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
            all_values.shape[0], -1, 2,
            all_values.shape[-1]).mean(2)
        
        # the mask of midpoint follows the mask of its two endpoints
        if mask_keep_part_in_new_pos is not None:
            edge_masks = mask_keep_part_in_new_pos[:, unique_edges.reshape(-1)].reshape(
                mask_keep_part_in_new_pos.shape[0], -1, 2
            )
            mid_points_mask = (edge_masks.sum(dim=2) == 2).float().unsqueeze(-1)
            new_mask = torch.cat([mask_keep_part_in_new_pos, mid_points_mask], 1)
        else:
            new_mask = None
        
        # position of: new verts and their midpoints
        new_v = torch.cat([all_values, mid_points_pos], 1)
        
        # get new tets, 8x subdivision within a tet
        idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
        idx_ab = idx_map[0::6]
        idx_ac = idx_map[1::6]
        idx_ad = idx_map[2::6]
        idx_bc = idx_map[3::6]
        idx_bd = idx_map[4::6]
        idx_cd = idx_map[5::6]
        
        tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
        tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
        tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
        tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
        tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
        tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
        tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
        tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)
        
        # 4 vert's indices of each tet
        tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
        tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
        tet = tet_np.long().to(device)
        
        F = tet_fx4.shape[0]
        parent_indices = torch.arange(F, device=device)
        sub_to_parent_idx = parent_indices.repeat(8)
        # print("tet shape: ", tet.shape)
        # print("sub_to_parent_idx shape: ", sub_to_parent_idx.shape)
    
        return new_v, tet, new_mask, sub_to_parent_idx