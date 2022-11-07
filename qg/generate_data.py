import os
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import numpy as np

from solver import generate_data


def generate_train(resolution, filename, num_funcs, num_linfuncs):
    # training data generation
    train_res = resolution
    num_linfun = num_linfuncs
    start_idx = 1
    num_cos = num_funcs

    f_funcs = []
    for i in range(start_idx, num_cos + start_idx):
        freq = i // 3 + 1
        phase = i % 11 + 1
        def func(xx, freq=freq, phase=phase):
            # print(freq)
            # print(phase)
            return np.cos(freq * xx + phase)
        f_funcs.append(func)

    for i in range(start_idx, num_linfun + start_idx):
        idx = i / num_linfun * 10
        def func(xx, A=idx):
            return A * xx

        f_funcs.append(func)

    xx, f, q, q_c, dq_c = generate_data(f_funcs=f_funcs, Nx=train_res)
    data_dict = {
        'xx': xx,
        'f': f,
        'q': q
    }
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Training data saved as {filename}')


def generate_test(resolution, filename, num_funcs):
    Ny = resolution
    # test data generation
    def f_func1(xx_test):
        return 6 * (1 - 2 * xx_test) ** 2 - 2 * (xx_test - xx_test ** 2) * (1 - 2 * xx_test) ** 2 + 2 * (
                    xx_test - xx_test ** 2) ** 2 + 2

    def f_func2(xx_test):
        f = np.ones_like(xx_test)
        f[xx_test <= 0.5] = 0.0
        f[xx_test > 0.5] = 10.0
        return f

    def f_func3(xx_test):
        L = 1
        return 10 * np.sin(2 * np.pi * xx_test / L)

    f_funcs = [f_func1, f_func2, f_func3]

    xx_test, f_test, q_test, q_c_test, dq_c_test = generate_data(f_funcs=f_funcs, Nx=resolution)

    data_dict = {'xx_test': xx_test, 'f_test': f_test, 'q_test': q_test}
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Test data saved at {filename}')


if __name__ == '__main__':
    parser = ArgumentParser('Parser for data generation')
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--res', type=int, default=100)
    parser.add_argument('--savedir', type=str, default='data/poisson')
    parser.add_argument('--num_funcs', type=int, default=40)
    args = parser.parse_args()

    # configuration
    base_dir = args.savedir
    os.makedirs(base_dir, exist_ok=True)

    filepath = os.path.join(base_dir, f'{args.name}-s{args.res}.pickle')
    if args.name == 'train':
        generate_train(resolution=args.res, filename=filepath, num_funcs=args.num_funcs, num_linfuncs=args.num_funcs)
    elif args.name == 'test':
        generate_test(resolution=args.res, filename=filepath, num_funcs=args.num_funcs)
    else:
        raise ValueError('--name should be either train or test')

