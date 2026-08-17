"""
Dataset class for raw multi-vehicle point clouds (no preprocess)
Supports ego_only for unsupervised learning
"""
from collections import OrderedDict
import numpy as np
import torch
import open3d as o3d
import os
import pickle


from opencood.data_utils.datasets import basedataset
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_ego_points, shuffle_points


class RawEarlyFusionDataset(basedataset.BaseDataset):
    """
    Return structure (per __getitem__):
    {
      'ego': {
        'merged_lidar': np.ndarray (N, C)   # all CAVs merged in ego frame; if ego_only, it's just ego lidar
        'cav_lidar': { cav_id: np.ndarray } # each CAV raw lidar (in ego frame)
        'cav_transforms': { cav_id: np.ndarray (4,4) } # T_cav->ego (for debugging/optional)
        'ego_id': cav_id_of_ego
      }
    }
    """
    def __init__(self, params, visualize=False, train=True, isSim=False,
                 ego_only=False, nofusion=False, single_pc=False,
                 cav_label_only=False):
        # ego_only/nofusion will be honored; isSim follows your upstream convention
        super(RawEarlyFusionDataset, self).__init__(
            params, visualize, train, isSim, ego_only, nofusion,
            single_pc, cav_label_only)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed = OrderedDict()
        processed['ego'] = {}

        # find ego vehicle's pose & id
        ego_id = -1
        ego_lidar_pose = None
        for cav_id, cav_content in base_data_dict.items():
            if cav_content.get('ego', False):
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert ego_id != -1 and ego_lidar_pose is not None

        cav_lidar_dict = {}
        cav_T_dict = {}
        cav_T_world_dict = {}
        projected_list = []
        object_stack = []

        # project each cav lidar into ego frame
        for cav_id, cav_content in base_data_dict.items():
            if self.ego_only and cav_id != ego_id:
                continue

            T_cav_to_ego = cav_content['params']['transformation_matrix']  # 4x4
            T_cav_to_world = cav_content['params']['lidar_pose']  # 4x4
            lidar_np = cav_content['lidar_np']                    # (N, C), C>=3

            # clean points
            lidar_np = mask_ego_points(lidar_np)
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(
                lidar_np[:, :3], T_cav_to_ego
            )

            cav_lidar_dict[cav_id] = lidar_np
            cav_T_dict[cav_id] = T_cav_to_ego
            cav_T_world_dict[cav_id] = T_cav_to_world
            projected_list.append(lidar_np)

            # === GT box 部分 ===
            if 'objects' in cav_content:  # 每个 cav_content 有 objects
                for obj in cav_content['objects']:
                    # obj 通常格式 (x,y,z,l,w,h,yaw)
                    box = np.array(obj['box']).reshape(1, 7)
                    # 转换到 ego 坐标系
                    box[:, :3] = box_utils.project_points_by_matrix_torch(
                        box[:, :3], T_cav_to_ego
                    )
                    object_stack.append(box)

        # merged lidar
        merged = np.vstack(projected_list) if len(projected_list) > 0 else np.zeros((0, 4), dtype=np.float32)

        # stack 所有 GT 框
        if len(object_stack) > 0:
            object_bbx_center = np.vstack(object_stack)  # (N, 7)
            object_bbx_mask = np.ones(object_bbx_center.shape[0])
        else:
            object_bbx_center = np.zeros((0, 7))
            object_bbx_mask = np.zeros((0,))

        processed['ego'].update({
            'merged_lidar': merged,
            'cav_lidar': cav_lidar_dict,
            'cav_transforms': cav_T_dict,
            'cav_poses': cav_T_world_dict,
            'ego_id': ego_id,
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask
        })

        if self.visualize:
            processed['ego']['origin_lidar'] = merged

        return processed

    def collate_batch(self, batch):
        """
        Generic collate for training with variable-size point clouds.
        Returns lists/dicts to avoid padding.
        """
        merged_list = []
        cav_lidar_list = []
        cav_T_list = []
        ego_id_list = []
        origin_lidar_list = []
        gt_boxes_list = []
        gt_masks_list = []

        for sample in batch:
            ego = sample['ego']
            merged_list.append(torch.from_numpy(ego['merged_lidar']).float())
            cav_lidar_list.append({k: torch.from_numpy(v).float() for k, v in ego['cav_lidar'].items()})
            cav_T_list.append({k: torch.from_numpy(v).float() for k, v in ego['cav_transforms'].items()})
            ego_id_list.append(ego['ego_id'])

            # ---- GT box 部分 ----
            if 'object_bbx_center' in ego:
                gt_boxes_list.append(torch.from_numpy(ego['object_bbx_center']).float())
                gt_masks_list.append(torch.from_numpy(ego['object_bbx_mask']).float())

            if self.visualize and 'origin_lidar' in ego:
                origin_lidar_list.append(torch.from_numpy(ego['origin_lidar']).float())

        out = {
            'merged_lidar': merged_list,      # len=B, each is (Ni, C)
            'cav_lidar': cav_lidar_list,      # len=B, dict per sample
            'cav_transforms': cav_T_list,     # len=B, dict per sample
            'ego_ids': ego_id_list
        }
        if len(gt_boxes_list) > 0:
            out['gt_boxes'] = gt_boxes_list   # len=B, each is (Mi, 7)
            out['gt_masks'] = gt_masks_list   # len=B, each is (Mi,)
        if self.visualize:
            out['origin_lidar'] = origin_lidar_list
        return out

    def collate_batch_test(self, batch):
        """
        Keep your previous constraint if you prefer test-only batch size==1.
        """
        assert len(batch) == 1, "Batch size 1 is required during testing!"
        sample = batch[0]['ego']
        out = {
            'merged_lidar': torch.from_numpy(sample['merged_lidar']).float(),
            'cav_lidar': {k: torch.from_numpy(v).float() for k, v in sample['cav_lidar'].items()},
            'cav_transforms': {k: torch.from_numpy(v).float() for k, v in sample['cav_transforms'].items()},
            'ego_id': sample['ego_id']
        }
        if 'object_bbx_center' in sample:
            out['gt_boxes'] = torch.from_numpy(sample['object_bbx_center']).float()
            out['gt_masks'] = torch.from_numpy(sample['object_bbx_mask']).float()
        if self.visualize and 'origin_lidar' in sample:
            out['origin_lidar'] = torch.from_numpy(sample['origin_lidar']).float()
        return out

    def visualize_sample(self, index, cav_id=None):
        """
        Visualize one sample's point cloud using Open3D.

        Parameters
        ----------
        index : int
            dataset index
        cav_id : int or None
            None = merged_lidar
            specific cav_id = that cav's lidar
        """
        sample = self.__getitem__(index)['ego']

        if cav_id is None:
            lidar_np = sample['merged_lidar']
            title = f"Index {index} - merged lidar"
        else:
            if cav_id not in sample['cav_lidar']:
                raise ValueError(f"CAV id {cav_id} not found in sample {index}")
            lidar_np = sample['cav_lidar'][cav_id]
            title = f"Index {index} - CAV {cav_id}"

        if lidar_np.shape[0] == 0:
            print(f"[Warning] No points to visualize for {title}")
            return

        # Only xyz for visualization
        pts = lidar_np[:, :3]

        # build open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # 可选：上色 merged vs cav
        if cav_id is None:
            pcd.paint_uniform_color([0.2, 0.6, 0.8])  # 蓝色
        else:
            pcd.paint_uniform_color([0.8, 0.3, 0.3])  # 红色

        print(f"Visualizing {title}, points={pts.shape[0]}")
        o3d.visualization.draw_geometries([pcd], window_name=title)
    
    def visualize_with_pseudo(self, index, pseudo_root, cav_id=None):
        """
        可视化点云 + 伪框 (pseudo boxes)

        Parameters
        ----------
        index : int
            dataset index
        pseudo_root : str
            存放伪标签的根目录，每帧对应一个 {index:06d}.pkl
        cav_id : int or None
            None = merged_lidar
            specific cav_id = that cav's lidar
        """
        sample = self.__getitem__(index)['ego']

        # --- 点云部分 ---
        if cav_id is None:
            lidar_np = sample['merged_lidar']
            title = f"Index {index} - merged lidar + pseudo boxes"
        else:
            if cav_id not in sample['cav_lidar']:
                raise ValueError(f"CAV id {cav_id} not found in sample {index}")
            lidar_np = sample['cav_lidar'][cav_id]
            title = f"Index {index} - CAV {cav_id} + pseudo boxes"

        pts = lidar_np[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色点云

        # --- 伪框部分 ---
        pseudo_file = os.path.join(pseudo_root, f"{index:06d}.pkl")
        geometries = [pcd]

        if os.path.exists(pseudo_file):
            with open(pseudo_file, "rb") as f:
                pseudo_dict = pickle.load(f)

            if "boxes" in pseudo_dict:
                print(pseudo_dict["boxes"].shape)
                pseudo_boxes = np.array(pseudo_dict["boxes"])[:, :7]

                for box in pseudo_boxes:
                    corners = box_utils.boxes_to_corners_3d(box.reshape(1, 7), 'lwh')[0]
                    lines = [
                        [0, 1], [1, 2], [2, 3], [3, 0],
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]
                    ]
                    colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色伪框
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(corners)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    geometries.append(line_set)

                print(f"[PseudoLabel] Index {index} | Loaded {len(pseudo_boxes)} pseudo boxes from {pseudo_file}")
            else:
                print(f"[PseudoLabel] Index {index} | No 'boxes' field in {pseudo_file}")
        else:
            print(f"[PseudoLabel] Index {index} | File not found: {pseudo_file}")

        o3d.visualization.draw_geometries(geometries, window_name=title)
