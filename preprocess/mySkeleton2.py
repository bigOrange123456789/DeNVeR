import cv2
import numpy as np
from skimage import img_as_bool
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from scipy.ndimage import map_coordinates
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from matplotlib import cm

def getGray(img_path = 'preprocess/data/test.png'):
    # 1. 读取彩色/灰度图
    # img_path = 'preprocess/data/test.png'                     # 换成你的文件名
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    return gray

def getCatheter(gray):
    # 2. 去噪（轻微高斯模糊）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 二值化：先 Otsu，再做形态学开/闭去噪
    # _, binary = cv2.threshold(blur, 0, 255,
    #                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. 转换为 0/1 的 bool 数组，满足 skeletonize 要求
    # binary_bool = img_as_bool(binary)

    binary_bool = blur > 0.1
    binary = blur
    binary[binary_bool] = 1
    binary[~binary_bool] = 0

    #################################### 一、获取骨架 ###################################

    # 5. 骨架提取
    skeleton = skeletonize(binary_bool)
    if False:# 6. 可视化（可选）
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.title('Binary'); plt.imshow(binary, cmap='gray')
        plt.subplot(1,2,2); plt.title('Skeleton'); plt.imshow(skeleton, cmap='gray')
        plt.tight_layout(); plt.show()

    #################################### 二、获取骨架上每个点的轴线方向 ###################################

    # 获取骨架点坐标
    import numpy as np
    y, x = np.where(skeleton)
    points = np.column_stack((x, y))  # shape: (N, 2)

    # 为每个点估计方向
    def estimate_tangent_directions(points, k=9):
        """
        对每个骨架点，用其k近邻拟合直线方向（切线方向）
        :param points: (N,2) ndarray
        :param k: 用于拟合的邻域点数
        :return: directions: (N,2) ndarray，单位方向向量
        """
        tree = KDTree(points, leaf_size=2)
        _, idx = tree.query(points, k=k)

        directions = []
        for i, ind in enumerate(idx):
            pts = points[ind]  # k×2
            # 中心化
            pts_centered = pts - pts.mean(axis=0)
            # SVD
            U, S, Vt = np.linalg.svd(pts_centered)
            direction = Vt[0]  # 主方向
            # 统一方向（可选：确保方向一致）
            if direction[0] < 0:
                direction = -direction
            directions.append(direction)
        return np.array(directions)

    directions = estimate_tangent_directions(points, k=9)

    if False:
        plt.figure(figsize=(6,6))
        plt.imshow(skeleton, cmap='gray')
        plt.quiver(points[:,0], points[:,1], directions[:,0], -directions[:,1],
                   color='r', scale=50, headwidth=3)
        plt.show()

    ################################### 三、局部半径（或叫 局部厚度） ###################################

    dirs = directions

    normals = np.stack((-dirs[:, 1], dirs[:, 0]), axis=1)   # (N,2)



    h, w = binary.shape
    max_len = max(h, w)

    radii = []
    for (cx, cy), n in zip(points, normals):
        t = np.arange(0, max_len, 0.5)

        line_fwd = np.column_stack((cx + t*n[0], cy + t*n[1]))
        mask_fwd = map_coordinates(binary.astype(float), line_fwd.T[::-1], order=0)
        mask_fwd_b = mask_fwd.astype(bool)
        len_fwd = np.where(mask_fwd_b == 0)[0][0] * 0.5 if np.any(mask_fwd_b == 0) else max_len

        line_bwd = np.column_stack((cx - t*n[0], cy - t*n[1]))
        mask_bwd = map_coordinates(binary.astype(float), line_bwd.T[::-1], order=0)
        mask_bwd_b = mask_bwd.astype(bool)
        len_bwd = np.where(mask_bwd_b == 0)[0][0] * 0.5 if np.any(mask_bwd_b == 0) else max_len

        radii.append(len_fwd + len_bwd)

    radii = np.array(radii)

    if False:
        radii[radii>25]=25
        plt.imshow(skeleton, cmap='gray')
        plt.scatter(points[:,0], points[:,1], c=radii, cmap='jet', s=5)#散点大小为 5 像素
        plt.colorbar(label='local radius (px)')
        plt.show()
    ################################### 四、基于“区域生长 + 半径突变阈值” 的骨架分段方案 ###################################
    # skeleton       : bool  2-D array   True=骨架
    # pts            : (N,2) array       骨架像素坐标
    # radii          : (N,) array        每点的局部半径
    pts = points

    G = nx.Graph()
    from sklearn.neighbors import NearestNeighbors
    # nn = NearestNeighbors(radius=1.5, metric='euclidean').fit(pts)
    nn = NearestNeighbors(radius=1.5, metric='euclidean').fit(pts)
    for i, p in enumerate(pts):
        G.add_node(i, pos=p, r=radii[i])
        neigh = nn.radius_neighbors([p], return_distance=False)[0]
        for j in neigh:
            if i != j:
                G.add_edge(i, j, weight=np.linalg.norm(pts[i]-pts[j]))

    ΔR_max = 2#2 #2#5#8#5.0#2.0           # 允许的最大半径差
    visited = np.zeros(len(pts), dtype=bool)#标记哪些骨架点已被访问，避免重复。
    segments = []
    # sk_idx_set = set(range(len(pts)))   # 所有骨架节点索引集合
    for seed in np.where(~visited)[0]:
        if visited[seed]:
            continue
        queue = [seed]
        segment = []
        while queue:
            cur = queue.pop(0)
            if visited[cur]:
                continue
            visited[cur] = True
            segment.append(cur)
            for nb in G[cur]:
                if not visited[nb] and abs(G.nodes[nb]['r']-G.nodes[cur]['r']) <= ΔR_max:
                # if not visited[nb] and nb in sk_idx_set and abs(G.nodes[nb]['r'] - G.nodes[cur]['r']) <= ΔR_max:
                    queue.append(nb)
        segments.append(np.array(segment))

    if False:
        import matplotlib.cm as cm
        colors = cm.get_cmap('gist_ncar', len(segments))  # 自动生成 100 种连续色
        colors = colors(np.arange(len(segments)))          # (100,4) RGBA

        plt.figure(figsize=(6,6))
        plt.imshow(skeleton, cmap='gray', vmin=0, vmax=1)

        for idx, c in zip(segments, colors):
            seg_pts = pts[idx]
            plt.scatter(seg_pts[:,0], seg_pts[:,1], c=c, cmap='jet', s=2)  # 散点大小为 5 像素

        plt.axis('off'); plt.gca().set_aspect('equal')
        plt.show()

    ################################### 五、找出导管 ###################################
    # 0.找出与图片边缘向量的线段
    edge_mask = np.zeros((h, w), dtype=bool)
    # edge_mask[[0, -1], :] = True  # 上下两行
    # edge_mask[:, [0, -1]] = True  # 左右两列
    for i in range(5):
        edge_mask[[i, -(i+1)], :] = True  # 上下两行
        edge_mask[:, [i, -(i+1)]] = True  # 左右两列
    edge_segments = []
    for seg in segments:
        # seg 里是骨架索引；取坐标
        seg_pts = pts[seg]  # (n,2) (x,y)
        x, y = seg_pts[:, 0], seg_pts[:, 1]
        hit = edge_mask[y, x].any()  # 有没有点在边缘
        if hit:
            edge_segments.append(seg)
    segments = edge_segments

    # 1. 设定阈值（像素单位，可根据需要调整）
    σ_max = 1.5   # 例：半径变化 < 1 px 的线段才保留

    # 2. 过滤
    filtered_segments = []
    for seg in segments:
        seg_r = radii[seg]          # 该线段所有半径
        pts_seg = pts[seg]

        if False:# 欧氏折线长度
            diffs = np.diff(pts_seg, axis=0)  # (m-1,2)
            length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:# 轴对齐包围框
            x_min, y_min = pts_seg.min(axis=0)
            x_max, y_max = pts_seg.max(axis=0)
            w = x_max - x_min
            h = y_max - y_min
            # perimeter = float(2 * (w + h))  # 包围框周长
            length = (w**2 + h**2)**0.5 # 包围框对角线长度

        # if length>150 and seg_r.std() < σ_max and seg_r.min() > seg_r.mean()*0.75 :
        if length > 0 and seg_r.std() < σ_max and seg_r.min() > seg_r.mean() * 0.5:
            filtered_segments.append(seg)

    # print(f"保留 {len(filtered_segments)} / {len(segments)} 条线段")

    # 3. 可视化
    if False:
        colors = cm.get_cmap('gist_ncar', len(filtered_segments))
        plt.figure(figsize=(6,6))
        plt.imshow(skeleton, cmap='gray', vmin=0, vmax=1)

        for idx, c in zip(filtered_segments, colors(range(len(filtered_segments)))):
            seg_pts = pts[idx]
            # plt.plot(seg_pts[:,0], seg_pts[:,1], color=c, linewidth=2)
            plt.scatter(seg_pts[:, 0], seg_pts[:, 1], c=c, cmap='jet', s=2)  # 散点大小为 5 像素

        plt.axis('off'); plt.gca().set_aspect('equal')
        plt.show()

    if False:
        h, w = skeleton.shape          # 原图尺寸
        masks = []                     # 每条线段一张 mask
        print("kaa")
        for idx, seg in enumerate(filtered_segments):
            print("idx",idx)
            # 创建黑底单通道图
            mask = np.zeros((h, w), dtype=np.uint8)
            # 把该线段所有骨架像素描成 255
            seg_pts = pts[seg]         # (n,2)
            mask[seg_pts[:, 1], seg_pts[:, 0]] = 255

            # （可选）把 1 像素骨架膨胀成条带，宽度可调
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)

            masks.append(mask)

            # masks[i] 就是第 i 条线段的二值 mask，可直接保存：
            cv2.imwrite(f'segment_{idx}.png', mask)

    ################################### 六、 每条线段的 mask（区域） ###################################
    '''
    binary	原始二值图 (H,W)，前景为 True / 255
    skeleton	骨架图 (H,W)，True 表示骨架像素
    segments	列表，每个元素是该线段在骨架上的像素索引数组
    pts	(N,2) ndarray，所有骨架像素坐标 (x,y)
    '''

    # # 2. 计算“最近骨架点属于哪条线段”
    # # 2.1 先把所有骨架像素按线段分好组，并给每条线段一个标签
    # import numpy as np
    # from sklearn.neighbors import NearestNeighbors
    #
    # # 给每条线段一个整数标签
    # labels = np.full(len(pts), -1, dtype=int)
    # for seg_id, seg in enumerate(segments):
    #     labels[seg] = seg_id          # seg 里是骨架索引
    #
    # #2.2 用 KD-Tree 快速找最近骨架点
    # # 所有骨架坐标
    # sk_pts = pts
    #
    # # 建立 KD-Tree
    # tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sk_pts)
    #
    # # 取出所有前景像素坐标
    # y, x = np.where(binary)
    # fg_pts = np.column_stack((x, y))   # (M,2)
    #
    # # 查询每个前景像素最近的骨架点索引
    # dist, idx = tree.kneighbors(fg_pts, return_distance=True)
    # nearest_seg_label = labels[idx.ravel()]   # shape=(M,)
    #
    # #3. 生成每段线段的专属 mask
    # h, w = binary.shape
    # masks = []                    # list of (H,W) bool arrays
    # for seg_id in range(len(segments)):
    #     mask = np.zeros((h, w), dtype=bool)
    #     # 把属于该 seg_id 的前景像素置 True
    #     mask[y[nearest_seg_label == seg_id],
    #          x[nearest_seg_label == seg_id]] = True
    #     masks.append(mask)
    # if False: #4. 可视化（可选）
    #     import matplotlib.pyplot as plt
    #     from matplotlib import cm
    #
    #     colors = cm.get_cmap('tab20', len(masks))
    #     plt.figure(figsize=(6,6))
    #     for m, c in zip(masks, colors(range(len(masks)))):
    #         plt.contour(m, colors=[c], linewidths=2)
    #     plt.imshow(binary, cmap='gray', alpha=0.3)
    #     plt.axis('off'); plt.show()

    # exit(0)
    ################################### 六.2、 线段的 mask（区域） ###################################

    '''
        binary            : 原始二值图 (H,W)
        pts               : 所有骨架像素坐标 (N,2)
        segments          : 全部分段（未过滤）
        filtered_segments : 过滤后留下的线段列表（里面的元素仍是索引数组）
    '''
    # 2. 建立“全局线段标签 → 保留标签”映射
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # 给所有骨架点打全局标签
    labels_all = np.full(len(pts), -1, dtype=int)
    for seg_id, seg in enumerate(segments):
        labels_all[seg] = seg_id

    # 只保留 filtered_segments 中的标签
    good_ids = {seg_id for seg_id, seg in enumerate(segments)
                # if seg in filtered_segments}
                if any(np.array_equal(seg, fs) for fs in filtered_segments)}

    # # 先拿到 filtered_segments 中每个线段在 segments 中的索引
    # filtered_indices = [i for i, seg in enumerate(segments)
    #                     if seg in filtered_segments]  # ❌ 会报错
    # # ✅ 正确写法：转成索引集合
    # filtered_indices = {i for i, seg in enumerate(segments)
    #                     if any(np.array_equal(seg, fs) for fs in filtered_segments)}

    # 3. KD-Tree：前景像素 → 最近骨架点 → 是否属于保留段
    # 所有前景像素坐标
    y, x = np.where(binary)
    fg_pts = np.column_stack((x, y))

    # KD-Tree 找最近骨架点
    tree = NearestNeighbors(n_neighbors=1).fit(pts)
    _, idx = tree.kneighbors(fg_pts)
    nearest_label = labels_all[idx.ravel()]        # 每条前景像素对应的线段 id
    keep = np.isin(nearest_label, list(good_ids))  # True/False 掩码

    # 4. 生成最终 mask
    h, w = binary.shape
    mask_filtered = np.zeros((h, w), dtype=bool)
    mask_filtered[y[keep], x[keep]] = True

    if False:# 5. 可视化 / 保存
        import matplotlib.pyplot as plt
        plt.imshow(mask_filtered, cmap='gray')
        plt.axis('off'); plt.show()


    return mask_filtered
def save(mask_filtered,path='mask_filtered.png'):
    # 保存
    import cv2
    cv2.imwrite(path, mask_filtered.astype(np.uint8) * 255)

if __name__ == "__main__":
    gray = getGray('preprocess/data/test.png') #gray <class 'numpy.ndarray'> (512, 512)
    mask_filtered = getCatheter(gray)
    save(mask_filtered, 'mask_filtered.png')
# if __name__ == "__main__":
#     pathIn = "gt_cath"
#     pathOut = "gt_cath2"
#     for file in os.listdir(pathIn):
#         gray = getGray(os.path.join(pathIn, file))
#         mask_filtered = getCatheter(gray)
#         save(mask_filtered, os.path.join(pathOut, file))