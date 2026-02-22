# coding = utf-8

import sys
import argparse

from util.experiment import Experiment
from util.conf import Configuration
import time

def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters for Indexing Embedding experiments')

    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    parser.add_argument('-E', '--embed', default=False, dest='to_embed', action='store_true', help='whether to embed database/query')

    args = parser.parse_args(argv[1: ])

    conf = Configuration(args.confpath, dump=True)

    if args.to_embed:
        conf.setHP('to_embed', True)

    experiment = Experiment(conf)
    experiment.run()

    # print(experiment.train_db_loader(0).shape)



if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv)
    # 记录程序结束时间
    end_time = time.time()

    # 计算并打印总运行时间（保留4位小数，更精准）
    total_time = end_time - start_time
    print(f"\n程序整体运行时间：{total_time:.4f} 秒")
