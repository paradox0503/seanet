# coding = utf-8

import sys
import argparse

from util.experiment import Experiment
from util.conf import Configuration
from util.multidata import prepare_datasets


def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters for Indexing Embedding experiments')
    
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    parser.add_argument('-E', '--embed', default=False, dest='to_embed', action='store_true', help='whether to embed database/query')

    args = parser.parse_args(argv[1: ])

    conf = Configuration(args.confpath)

    if args.to_embed:
        conf.setHP('to_embed', True)
        if conf.getHP('datasets'):
            conf.setHP('datasets', prepare_datasets(conf.getHP('datasets'),
                       conf.getHP('size_train'), conf.getHP('size_val'), True))

    conf.dumpConf()

    experiment = Experiment(conf)
    experiment.run()


if __name__ == "__main__":
    main(sys.argv)
