from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils.file_io import PathManager
from .bases import ImageDataset

@DATASET_REGISTRY.register()
class CUHKNP(ImageDataset):

    dataset_name = "cuhknp"
    # 咋不检查？？
    def __init__(self, root, **kwargs):
        self.root = osp.join(root, 'cuhknp')
        self.root = osp.join(self.root, 'detected')
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
        self.num_train_pids, self.num_query_pids, self.num_gallery_pids = 0, 0, 0
        self.has_time_info = False

        train, self.num_train_pids = self.preprocess(self.train_path)
        gallery, self.num_gallery_pids = self.preprocess(self.gallery_path, False)
        query, self.num_query_pids = self.preprocess(self.query_path, False)
        
        super(CUHKNP, self).__init__(train, query, gallery, **kwargs)

    @property
    def images_dir(self):
        return None

    def preprocess(self, path, is_train=True):
        # fname = '0003_c1_22.png'
        # pattern = 3 1
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []

        # 找到路径下，所有 *.png 图像
        fpaths = sorted(glob(osp.join(self.root, path, '*.png')))

        for fpath in fpaths:
            # 加载最右的路径 即 0003_c1_22.png
            fname = osp.basename(fpath)
            # pattern.search(string, pos, endpos).group(), 令string在pos和endpos区间内扫描字符, 不指定
            # pos和endpos则扫描整个string，并最后返回和pattern匹配的对象
            pid, camid = map(int, pattern.search(fname).groups())
            # CUHK03-NP有很多垃圾数据，是单纯的背景，id为-1，若遇到直接跳过
            if pid == -1: continue
            if is_train:
                # 获得所有的 pid，本质是去重，从0开始
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            # 去不去重，都得到 pid
            pid = all_pids[pid]
            # 把camera id处理成从 0 开始
            camid -= 1
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            ret.append((fpath, pid, camid))
        return ret, int(len(all_pids))
