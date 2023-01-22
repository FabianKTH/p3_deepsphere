def nocoeff_from_l(l_max):
    return (l_max + 1) ** 2


def _antipod_lmax(l_max):
    assert not l_max % 2, f'l_max {l_max} not even!'

    idx_list = list()
    m = (l_max + 1) * (l_max // 2 + 1)
    n = (l_max + 1) ** 2

    even_idx = list(range(m))
    odd_idx = list(range(m, n))

    # print(f'l_max: {l_max}')
    # print(f'm: {m}')

    for l in range(l_max + 1):
        k = 2 * l + 1
        # n = (l)**2

        # print(idx_list)

        if not l % 2:
            # even
            # l_even += 1
            # idx_list.extend(list(range( int(n), int(n+k) )))
            for _ in range(k):
                idx_list.append(even_idx.pop(0))
        else:
            # l_odd += 1
            # idx_list.extend(list(range( int(n+m), int(n+m+k) )))
            for _ in range(k):
                idx_list.append(odd_idx.pop(0))

    return idx_list


def _init_antipod_dict(l_max=8):
    assert not l_max % 2, f'l_max {l_max} not even!'

    _antipod_dict = dict()
    for l in range(0, l_max + 1, 2):
        _antipod_dict[l] = _antipod_lmax(l)

    return _antipod_dict