import numpy as np

def show_cluster_counters(labels, cnt_per_row=4, show_detailed=True, singular_threshold=1):
    unique_labels, cnt = np.unique(labels, return_counts=True)
    idx = 0
    if show_detailed:
        for (l, c) in zip(unique_labels, cnt):
            print("num of ", l, ":", c, end=",")
            if idx % cnt_per_row == 0:
                print()
            idx = idx + 1
    singular_cnt = np.sum(cnt[cnt <= singular_threshold])
    print("sigular clusters number", singular_cnt)
    print("singular percent", singular_cnt / np.sum(cnt))
    print("max number in a cluster : ", np.max(cnt))


