import os
import argparse
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims_ae1', type=int, nargs='+',
                        default=[1024, 1000, 500, 100], help='dims_ae1')
    parser.add_argument('--dims_ae2', type=int, nargs='+',
                        default=[944, 1000, 500, 100], help='dims_ae2')
    parser.add_argument('--dims_ae3', type=int, nargs='+',
                        default=[4096, 2000, 1000, 100], help='dims_ae3')
    parser.add_argument('--dims_ae1_graph', type=int, nargs='+',
                        default=[1440, 1500, 750, 100], help='dims_ae1_graph')
    parser.add_argument('--dims_ae2_graph', type=int, nargs='+',
                        default=[1440, 1500, 750, 100], help='dims_ae2_graph')
    parser.add_argument('--dims_ae3_graph', type=int, nargs='+',
                        default=[1440, 1500, 750, 100], help='dims_ae3_graph')
    parser.add_argument('--act', type=str,
                        default='tanh', help='activation function')
    parser.add_argument('--batch_size', type=int,
                        default=100, help='batch_size')
    parser.add_argument('--epochs_pre', type=int, default=20,
                        help='num of pre epochs')
    parser.add_argument('--epochs_fine', type=int, default=20,
                        help='number of finetune epochs')
    parser.add_argument('--lr_pre', type=float,
                        default=1.0e-3, help='learning rate')
    parser.add_argument('--lr_fine', type=float,
                        default=1.0e-3, help='learning rate')
    parser.add_argument('--lr_joint', type=float,
                        default=1.0e-6, help='learning rate')
    parser.add_argument('--gamma', type=int,
                        default=13, help='bandwidth parameter')
    parser.add_argument('--k', type=int,
                        default=35, help='number of neighbor nodes')
    parser.add_argument('--alpha', type=float,
                        default=0.01, help='balance parameter')
    parser.add_argument('--beta', type=float,
                        default=0.001, help='balance parameter')

    args = parser.parse_args()
    return args