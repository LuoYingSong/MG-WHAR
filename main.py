import argparse

from model.mgwhar import ANGELO, para_dict
from util.trainer import test, device
from util.util import matrixLoader2, count_parameters, set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('-s', '--seed', type=int, default=2022, help='the placement of sensors')
    parser.add_argument('-w', '--word', type=int, default=64, help='the placement of sensors')
    parser.add_argument('-t', '--topic', type=int, default=64, help='the placement of sensors')
    args = parser.parse_args()
    set_seed(args.seed)
    body_graph = matrixLoader2('data/dataset/opportunity/body.txt')
    sensor_graph = matrixLoader2('data/dataset/opportunity/type.txt')
    pattern_graph = matrixLoader2(
        f'data/dataset/opportunity/opportunity_5imu/pattern_graph_{args.word}_{args.topic}.txt')
    model = ANGELO(g1=body_graph, g2=sensor_graph, g3=pattern_graph, dataset=f'opportunity').to(device=device)
    config_train = {'batch_size': 128,
                    'optimizer': 'Adam',
                    'lr': 1e-3,
                    'lr_step': 10,
                    'lr_decay': 0.9,
                    'init_weights': 'orthogonal',
                    'epochs': 1,
                    'print_freq': 100
                    }
    test_config = {'batch_size': 1,
                   'train_mode': False,
                   'num_batches_eval': 212}
    count_parameters(model, [1, 15, 24, 3])
    acc_test, fm_test, fw_test, elapsed, loss = test(model, 24, 'data/dataset/opportunity/opportunity_5imu',
                                                     f'data/logs/opportunity', para_dict, para_dict, mode='restart',
                                                     show_test=False, word=args.word, topic=args.topic)
