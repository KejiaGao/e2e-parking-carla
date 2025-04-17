import torch
import math

from torch import nn
from model.cam_encoder import CamEncoder
from tool.config import Configuration
from tool.geometry import VoxelsSumming, calculate_birds_eye_view_parameters


class BevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        bev_res, bev_start_pos, bev_dim = calculate_birds_eye_view_parameters(self.cfg.bev_x_bound,
                                                                              self.cfg.bev_y_bound,
                                                                              self.cfg.bev_z_bound)
        self.bev_res = nn.Parameter(bev_res, requires_grad=False)
        self.bev_start_pos = nn.Parameter(bev_start_pos, requires_grad=False)
        self.bev_dim = nn.Parameter(bev_dim, requires_grad=False)

        self.down_sample = self.cfg.bev_down_sample

        self.frustum = self.create_frustum()
        self.depth_channel, _, _, _ = self.frustum.shape
        self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)


    def create_frustum(self):
        h, w = self.cfg.final_dim
        down_sample_h, down_sample_w = h // self.down_sample, w // self.down_sample

        depth_grid = torch.arange(*self.cfg.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, down_sample_h, down_sample_w)
        depth_slice = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, down_sample_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, down_sample_w).expand(depth_slice, down_sample_h, down_sample_w)
        y_grid = torch.linspace(0, h - 1, down_sample_h, dtype=torch.float)
        y_grid = y_grid.view(1, down_sample_h, 1).expand(depth_slice, down_sample_h, down_sample_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinsics, extrinsics):
        extrinsics = torch.inverse(extrinsics).cuda()
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        b, n, _ = translation.shape

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        combine_transform = rotation.matmul(torch.inverse(intrinsics)).cuda()
        points = combine_transform.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(b, n, 1, 1, 1, 3)

        return points

    def encoder_forward(self, images):
        """
        This function processes multi-frame images using an attention mechanism to fuse temporal information.
        The input images are expected to have shape [B, T, N, C, H, W], where:
            B = batch size,
            T = number of time frames (e.g. [current, current-5, current-10]),
            N = number of camera views,
            C, H, W = channel, height, and width.
        
        The function returns fused features with shape [B, N, D, H', W', C] (same as the single-frame output)
        and the depth probability (depth_prob).
        """
        # print("images.shape:", images.shape) # original[5, 4, 3, 256, 256] vs channel concat[5, 3, 4, 3, 256, 256]
        B, T, N, C, H, W = images.shape

        # Merge the time and camera dimensions to process all images with the CamEncoder
        images = images.view(B * T * N, C, H, W)
        
        # Get image features (x) and depth predictions from the camera encoder
        x, depth = self.cam_encoder(images)
        
        # Compute depth probability distribution along the depth dimension
        depth_prob = depth.softmax(dim=1)
        
        # Multiply image features with depth probabilities if using depth distribution,
        # otherwise repeat the feature along the depth channel.
        if self.cfg.use_depth_distribution:
            # x: [B*T*N, feat_channels, H', W']
            # depth_prob: [B*T*N, D, H', W']
            # Unsqueeze to enable outer product -> result: [B*T*N, feat_channels, D, H', W']
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2)
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channel, 1, 1)
        
        # Reshape the features back to separate the batch, time, and camera dimensions.
        # x now has shape: [B, T, N, feat_channels, D, H', W']
        x = x.view(B, T, N, *x.shape[1:])
        
        # Permute dimensions to have depth, height, and width before the channel:
        # New shape: [B, T, N, D, H', W', feat_channels]
        x = x.permute(0, 1, 2, 4, 5, 6, 3)
        
        # ---------------- Temporal Fusion with Attention ----------------
        # We now have multi-frame features with shape: [B, T, N, D, H', W', C]
        # For each camera view and for each spatial location in the BEV space (D, H', W'),
        # we want to fuse the T frames, giving more weight to the current frame (assumed at index 0).
        
        # First, permute to bring the camera and spatial dimensions together:
        # New shape: [B, N, D, H', W', T, C]
        x = x.permute(0, 2, 3, 4, 5, 1, 6)
        B, N, D, H_prime, W_prime, T, C = x.shape
        
        # Flatten all dimensions except the temporal dimension.
        # New shape: [B * N * D * H' * W', T, C]
        x_reshaped = x.reshape(B * N * D * H_prime * W_prime, T, C)
        
        # Use the current frame (time index 0) as the query.
        # Shape of query: [B*N*D*H'*W', 1, C]
        query = x_reshaped[:, 0:1, :]
        
        # Use all frames as keys and values.
        key = x_reshaped   # Shape: [B*N*D*H'*W', T, C]
        value = x_reshaped # Shape: [B*N*D*H'*W', T, C]
        
        # Compute scaled dot-product attention scores.
        # The scores tensor has shape: [B*N*D*H'*W', 1, T]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(C)
        
        # Apply softmax to get attention weights along the time dimension.
        attn_weights = torch.softmax(scores, dim=-1)  # Shape: [B*N*D*H'*W', 1, T]
        
        # Fuse the temporal features by computing the weighted sum.
        # The resulting fused feature has shape: [B*N*D*H'*W', 1, C]
        fused_feature = torch.matmul(attn_weights, value)
        fused_feature = fused_feature.squeeze(1)  # Now shape: [B*N*D*H'*W', C]
        
        # Reshape back to the expected BEV feature shape: [B, N, D, H', W', C]
        fused_feature = fused_feature.view(B, N, D, H_prime, W_prime, C)
        
        # --------------------------------------------------------------------
        # Return the temporally fused BEV feature and the depth probability.
        return fused_feature, depth_prob


    # def encoder_forward(self, images): # original
    #     print("images.shape:", images.shape) # 4-view concat[5, 3, 3, 512, 768] vs original[5, 4, 3, 256, 256] vs channel concat[5, 3, 4, 3, 256, 256]
    #     b, n, c, h, w = images.shape
    #     images = images.view(b * n, c, h, w)
    #     x, depth = self.cam_encoder(images)
    #     depth_prob = depth.softmax(dim=1)
    #     if self.cfg.use_depth_distribution:
    #         x = depth_prob.unsqueeze(1) * x.unsqueeze(2)
    #     else:
    #         x = x.unsqueeze(2).repeat(1, 1, self.depth_channel, 1, 1)
    #     x = x.view(b, n, *x.shape[1:])
    #     x = x.permute(0, 1, 3, 4, 5, 2)
    #     return x, depth_prob

    def proj_bev_feature(self, geom, image_feature):
        # print("self.cfg.d_bound:", self.cfg.d_bound)
        # print("depth_channel from frustum:", self.frustum.shape[0])
        # print("depth_channel from image_feature:", image_feature.shape[2])
        # print(image_feature.size()) # 4-view concat [5, 3, 48, 64, 96, 64]; channel concat [5, 4, 48, 32, 32, 64]
        batch, n, d, h, w, c = image_feature.shape
        output = torch.zeros((batch, c, self.bev_dim[0], self.bev_dim[1]),
                             dtype=torch.float, device=image_feature.device)
        N = n * d * h * w # 4-view concat 884736; channel concat 196608
        for b in range(batch):
            image_feature_b = image_feature[b]
            geom_b = geom[b]

            x_b = image_feature_b.reshape(N, c)
            # print(x_b.size()) # [884736, 64]

            geom_b = ((geom_b - (self.bev_start_pos - self.bev_res / 2.0)) / self.bev_res)
            # print(geom_b.size()) # 4-view concat [4, 48, 64, 96, 3]; channel concat [4, 48, 32, 32, 3]
            geom_b = geom_b.view(N, 3).long()

            mask = ((geom_b[:, 0] >= 0) & (geom_b[:, 0] < self.bev_dim[0])
                    & (geom_b[:, 1] >= 0) & (geom_b[:, 1] < self.bev_dim[1])
                    & (geom_b[:, 2] >= 0) & (geom_b[:, 2] < self.bev_dim[2]))
            x_b = x_b[mask]
            geom_b = geom_b[mask]

            ranks = ((geom_b[:, 0] * (self.bev_dim[1] * self.bev_dim[2])
                     + geom_b[:, 1] * self.bev_dim[2]) + geom_b[:, 2])
            sorts = ranks.argsort()
            x_b, geom_b, ranks = x_b[sorts], geom_b[sorts], ranks[sorts]

            x_b, geom_b = VoxelsSumming.apply(x_b, geom_b, ranks)

            bev_feature = torch.zeros((self.bev_dim[2], self.bev_dim[0], self.bev_dim[1], c),
                                      device=image_feature_b.device)
            bev_feature[geom_b[:, 2], geom_b[:, 0], geom_b[:, 1]] = x_b
            tmp_bev_feature = bev_feature.permute((0, 3, 1, 2)).squeeze(0)
            output[b] = tmp_bev_feature

        return output

    def calc_bev_feature(self, images, intrinsics, extrinsics):
        # print("images.size():", images.size())
        geom = self.get_geometry(intrinsics, extrinsics)
        x, pred_depth = self.encoder_forward(images)
        bev_feature = self.proj_bev_feature(geom, x)
        return bev_feature, pred_depth

    def forward(self, images, intrinsics, extrinsics):
        bev_feature, pred_depth = self.calc_bev_feature(images, intrinsics, extrinsics)
        return bev_feature.squeeze(1), pred_depth

