# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image

"""
class CommDataset(Dataset):
    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        dom_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
            if relabel:
                dom_set.add(i[1].split('_')[0])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.doms = {}
        for idx, dom in enumerate(list(dom_set)):
            self.doms[dom] = idx
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            self.dom_dict = {}
            for pid in self.pids:
                did = pid.split('_')[0]
                self.dom_dict[pid] = self.doms[did]

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        domid = [] if len(self.doms.values()) == 0 else list(self.dom_dict.values())

        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            domid = self.dom_dict[pid]
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "domains": domid,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
"""

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.pid_expert_dict = dict()
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            for pid in self.pids:
                dataset, pid_exper = pid.split('_')[0], pid.split('_')[-1]
                if dataset not in self.pid_expert_dict.keys(): self.pid_expert_dict[dataset] = dict()
                self.pid_expert_dict[dataset][pid_exper] = len(self.pid_expert_dict[dataset].keys())
    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)

        #pdb.set_trace()
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid_agg = self.pid_dict[pid]
            camid = self.cam_dict[camid]
            pid_expert = int(pid.split('_')[-1])
            return {
                "images": img,
                "targets": pid_agg,
                "targets_expert": self.pid_expert_dict[pid.split('_')[0]][pid.split('_')[-1]],
                "camids": camid,
                "img_paths": img_path,
            }
        else:
            return {
                "images": img,
                "targets": pid,
                "camids": camid,
                "img_paths": img_path,
            }  

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

