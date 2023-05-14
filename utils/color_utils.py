import math
import json
from collections import namedtuple
from typing import List


ColorCode = namedtuple('ColorCode', ['id', 'name', 'color'])


def save_color_codes(color_codes: List[ColorCode], fpath: str):
    for cc in color_codes:
        assert isinstance(cc.id, int), 'Color ID must be integer so it can be sorted.'
        for c in cc.color:
            assert isinstance(c, int), 'RGB color code must be integer.'
            assert 0<= c <= 255, 'RGB color code must set in range [0, 255].'
    dicts = [cc._asdict() for cc in color_codes]
    with open(fpath, 'w') as f:
        json.dump(dicts, f)


def load_color_codes(fpath: str, ordered=True):
    with open(fpath, 'r') as f:
        dicts = json.load(f)
    if ordered:
        dicts = sorted(dicts, key=lambda x: x['id'])
    return dicts


def generate_colors(num_labels):
    # Compute the number of different R, G, B values
    m = int(math.ceil(num_labels ** (1/3)))
    p = int(math.ceil(num_labels / m) ** 0.5)
    q = int(math.ceil(num_labels / (m * p)))

    # Compute the step size for each color channel
    step_size = int(256 / max(m, p, q))

    # Create a list of RGB colors
    colors = []
    for i in range(m):
        for j in range(p):
            for k in range(q):
                # Compute the current RGB color by multiplying the step size by the index
                color = (i * step_size, j * step_size, k * step_size)
                colors.append(color)

    return colors[:num_labels]


if __name__ == '__main__':
    colors = generate_colors(128)
    print(len(colors))
    for c in colors:
        print(c)