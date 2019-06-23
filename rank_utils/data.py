import numpy as np
import mxnet as mx





def list2pairs(list_x, list_y, n_samples=-1):
    """
    Given a list (features and scores) generate `n_samples` pairwise samples
    :param list_x: list of features, size (n_sample, n_features)
    :param list_y: list of scores / target, the larger the better, size (n_sample, ) or (n_sample, 1)
    :param n_samples: specifies how many pairs to generate from the list, <=0 means exhuastive generation
    :return: a list of pairwise data in the format of [[x_i, x_j, y_i, y_j], ...]
    """
    assert len(list_x) == len(list_y)
    if isinstance(list_y[0], list):
        list_y_element_is_list = True
    else:
        list_y_element_is_list = False

    list_len = len(list_y)
    if n_samples > 0:
        pairs_ids = np.random.randint(0, list_len, (n_samples, 2))
    else:
        pairs_ids = []
        for i in range(list_len):
            for j in range(i + 1, list_len):
                if i != j:
                    pairs_ids.append([i, j])
    pairs_data = []
    for i, j in pairs_ids:
        if list_y_element_is_list:
            pairs_data.append([list_x[i], list_x[j], list_y[i], list_y[j]])
        else:
            pairs_data.append([list_x[i], list_x[j], [list_y[i]], [list_y[j]]])

    return pairs_data


def batcify_func(examples):
    # TODO: make this more concise
    x_i = []
    x_j = []
    t_i = []
    t_j = []

    for e in examples:
        x_i.append(e[0])
        x_j.append(e[1])
        t_i.append(e[2])
        t_j.append(e[3])

    x_i = mx.nd.array(x_i)
    x_j = mx.nd.array(x_j)
    t_i = mx.nd.array(t_i)
    t_j = mx.nd.array(t_j)

    batch = [x_i, x_j, t_i, t_j]

    return batch
