import dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import util

def make_batch(samples):

    filenames = [sample['filename'] for sample in samples]
    coordinates = [sample['utm_coordinate'] for sample in samples]
    timestamps = [sample['timestamp'] for sample in samples]
    images = [sample['image'] for sample in samples]

    ret = {
        'filename' : filenames,
        'utm_coordinate' : coordinates,
        'timestamp' : timestamps,
        'image' : images
    }

    if 'pos' in samples[0].keys():
        ret['pos'] = [sample['pos'] for sample in samples]
        ret['neg'] = [sample['neg'] for sample in samples]

    return ret


class TokyoTrainDataLoader(DataLoader):

    def __init__(self, mode, batch_size=4, collate_fn=make_batch, db_loader=None):
        self.data_set = dataset.TokyoTrainDataSet(mode=mode)
        super().__init__(self.data_set, batch_size=batch_size,
                         collate_fn=collate_fn
                         )
        if db_loader is not None:
            util.query_pos_neg(db_loader, self)


class TokyoValDataLoader(DataLoader):

    def __init__(self, mode, batch_size=4, collate_fn=make_batch, db_loader=None):
        self.data_set = dataset.TokyoValDataSet(mode=mode)
        super().__init__(self.data_set, batch_size=batch_size,
                         collate_fn=collate_fn
                         )
        if db_loader is not None:
            util.query_pos_neg(db_loader, self)


class Tokyo247DataLoader(DataLoader):

    def __init__(self, mode, batch_size=4, collate_fn=make_batch):
        self.data_set = dataset.Tokyo247(mode=mode)
        super().__init__(self.data_set, batch_size=batch_size,
                         collate_fn=collate_fn
                         )
