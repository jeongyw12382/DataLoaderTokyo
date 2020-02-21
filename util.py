import geopy.distance
import utm
import numpy as np
import tqdm
import time

from sklearn.metrics.pairwise import euclidean_distances

# 54, 'S' because all the distance is calculated inside the Tokyo

posThr = 10.0
negThr = 25.0

def utm_distance(utm1, utm2):
    coord1 = utm.to_latlon(utm1[0], utm1[1], 54, 'S')
    coord2 = utm.to_latlon(utm2[0], utm2[1], 54, 'S')
    return geopy.distance.distance(coord1, coord2).m

def query_pos_neg(dbloader, queryloader, pos_threshold=posThr, neg_threshold=negThr):

    dbset = dbloader.dataset
    queryset = queryloader.dataset

    dbcoord = dbset.utm
    qcoord = queryset.utm

    dist = euclidean_distances(qcoord, dbcoord)
    pos = []
    neg = []
    for mat in dist > neg_threshold:
        neg.append(np.random.choice(np.argwhere(mat).T[0], 20))
    for i, mat in enumerate(dist < pos_threshold):
        candidates = np.random.permutation(np.argwhere(mat).T[0])
        val = dist[i][candidates]
        pos.append(candidates[np.argsort(val)[:20]])

    for i in range(len(pos)):
        queryset.set(i, 'pos', pos[i])
    for i in range(len(neg)):
        queryset.set(i, 'neg', neg[i])

if __name__=='__main__':
    import dataloader
    a = dataloader.TokyoTrainDataLoader(mode='db')
    print(a.get_image(3))