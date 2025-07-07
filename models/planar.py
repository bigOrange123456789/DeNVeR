import torch
import torch.nn as nn

import sys

sys.path.append("..")
import utils


class PlanarMotion(nn.Module): #建模平面运动（仿射/透视变换）#每个平面整体运动
    def __init__(self, n_frames, n_layers, scale=None, trans=None, **kwargs):
        """
            功能：建模平面运动（仿射/透视变换）
            n_frames #(int) N   #总帧数        #5 #视频中关键帧的数量(这里是每2帧取一个关键帧)
            n_layers #(int) M   #层数         #1
            scale    #(N, M, 2) #各层的初始缩放 #None
            trans    #(N, M, 2) #各层的初始平移 #None
        """
        super().__init__()
        self.n_frames = n_frames
        self.n_layers = n_layers
        N, M = n_frames, n_layers

        if scale is None: # 为空的时候表示放缩比例为1
            scale = torch.ones(N, M, 2, dtype=torch.float32)
        if trans is None: # 为空的时候表示移动距离为0
            trans = torch.zeros(N, M, 2, dtype=torch.float32)

        ## initialize zeros for the rotation and skew effects
        ## 8 total parameters:
        ## first four for sim transform
        ## second two extend to affine
        ## last two extend to perspective
        '''
        theta 存储8个运动参数（可学习）： #我只能理解前四个参数
            a, b：旋转/剪切参数
            tx, ty：平移参数
           -------------------
            k：纵横比
            w：剪切参数
           -------------------
            v1, v2：透视参数
        '''
        init = torch.zeros(N, M, 8, dtype=torch.float32) # 每个张图片8个参数(一个3*3的变换矩阵)
        self.register_parameter("theta", nn.Parameter(init, requires_grad=True)) #将可学习的参数记录为theta参数
        print(f"Initialized planar motion with {N} params, {M} layers")

        self.update_scale(scale)
        self.update_trans(trans)

    def forward(self, idx, grid):
        """
        :param idx (B) which transform to evaluate
        :param grid (B, M, H, W, 3) query grid in view
        :returns cano coordinates of query grid (B, M, H, W, 2)
        -------------------------------------------------------
        输入：
            idx：帧索引（形状 (B)）
            grid：查询网格（形状 (B, M, H, W, 3)） #像素坐标, v_pos #(x,y,1)
        输出：
            变换后的规范坐标（形状 (B, M, H, W, 2)）
        处理流畅：
            1.通过idx获取运动参数 theta #[9, 1, 8]
            2.转换为3x3变换矩阵 mat     #[9, 1, 3, 3]
            3.应用齐次变换到网格 grid
        """
        # idx: tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        # grid: type=<Tensor> shape=[9, 1, 128, 128, 3] #这个查询网格为啥是3通道的？
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return utils.apply_homography_xy1(mat, grid)

    def get_cano2view(self, idx): #返回从规范空间到视图空间的逆变换
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return torch.linalg.inv(mat)

    def get_theta(self, idx):
        """
        :param idx (B) which transforms to index
        :returns (B, M, 8)
        """
        return self.theta[idx]

    def update_scale(self, scale):
        """
        :param scale (N, M, 2)
        """
        with torch.no_grad():
            sx = scale[..., 0:1]
            sy = scale[..., 1:2]
            s = torch.sqrt(sx * sy)
            k = torch.sqrt(sx / sy)
            self.theta[..., 0:1] = s
            self.theta[..., 4:5] = k
            print("updated scale")

    def update_trans(self, trans): #修改所有关键点的tx, ty这两个平移参数
        """
        :param trans (N, M, 2)
        """
        with torch.no_grad():
            self.theta[..., 2:4] = trans
            print("updated trans")

    def theta_to_mat(self, theta): #参数转矩阵
        """
        expands the 8 parameters into 3x3 matrix
        H = [[A, t], [v.T, 1]] where A = SK + tv.T
        :param theta (N, M, 8)
        :returns mat (N, M, 3, 3)
        """
        *dims, D = theta.shape
        a = theta[..., 0:1]# 旋转/缩放
        b = theta[..., 1:2]# 旋转/剪切
        t = theta[..., 2:4, None]  # (*, 2, 1) # 平移 (..., 2, 1)
        k = theta[..., 4:5] + 1e-6 # 纵横比
        w = theta[..., 5:6]        # 剪切
        vT = theta[..., None, 6:8]  # (*, 1, 2) # 透视

        # 构建SK矩阵（旋转+缩放+剪切）
        SK = torch.cat([a * k, a * w + b / k, -b * k, -b * w + a / k], dim=-1).reshape(
            *dims, 2, 2
        )

        # 计算仿射部分: A = SK + t * vT
        A = SK + torch.matmul(t, vT)
        '''
        H = [
          [ SK + t·vT   |   t   ]
          [     vT      |   1   ]
        ]
        --------------------------
        a, b, k, w：控制旋转、缩放、剪切（仿射变换）
        t：控制平移
        vT：控制透视变换
        '''

        # 构建完整3x3矩阵
        return torch.cat(
            [
                torch.cat([A, t], dim=-1),
                torch.cat([vT, torch.ones_like(vT[..., :1])], dim=-1),
            ],
            dim=-2,
        )

    def get_scale(self, idx):
        """
        :param idx (B) which transforms to select
        :returns scale (B, M) of the transform
        """
        theta = self.get_theta(idx)  # (B, M, 8)
        return torch.sqrt(theta[..., 0] ** 2 + theta[..., 1] ** 2)  # (B, M)


class PlanarMotionNaive(nn.Module):
    def __init__(self, n_frames, n_layers, scales=None, trans=None, **kwargs):
        """
        :param n_frames (int) N total number of frames to model
        :param n_layers (int) M number of layers
        :param scales (N, M, 2) the initial scale of each layer
        :param trans (N, M, 2) the initial translation of each layer
        """
        super().__init__()
        self.n_frames = n_frames
        self.n_layers = n_layers
        N, M = n_frames, n_layers

        if scales is None:
            scales = torch.ones(N, M, 2, dtype=torch.float32)
        if trans is None:
            trans = torch.zeros(N, M, 2, dtype=torch.float32)

        sx = scales[..., 0:1]
        sy = scales[..., 1:2]
        tx = trans[..., 0:1]
        ty = trans[..., 1:2]
        z = torch.zeros_like(sx)

        init = torch.cat([sx, z, tx, z, sy, ty, z, z], dim=-1)
        self.register_parameter("theta", nn.Parameter(init, requires_grad=True))

    def forward(self, idx, grid):
        """
        :param idx (B) which transform to evaluate
        :param grid (B, M, H, W, 3) query grid in view
        :returns cano coordinates of query grid (B, M, H, W, 2)
        """
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return utils.apply_homography_xy1(mat, grid)

    def get_cano2view(self, idx):
        mat = self.theta_to_mat(self.get_theta(idx))  # (B, M, 3, 3)
        return torch.linalg.inv(mat)

    def get_theta(self, idx):
        """
        :param idx (B) which transforms to index
        :returns (B, M, 8)
        """
        return self.theta[idx]

    def theta_to_mat(self, theta):
        """
        expands the 8 parameters into 3x3 matrix
        :param theta (N, M, 8)
        :returns mat (N, M, 3, 3)
        """
        *dims, D = theta.shape
        ones = torch.ones_like(theta[..., :1])
        return torch.cat([theta, ones], dim=-1).view(*dims, 3, 3)

    def get_scale(self, idx):
        """
        :param idx (B) which transforms to select
        :returns scale (B, M) of the transform
        """
        theta = self.get_theta(idx)  # (B, M, 8)
        return 0.5 * (theta[..., 0] + theta[..., 2])  # (B, M)
