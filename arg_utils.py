from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('batch', default=16, type=int, help=' ')
    return parser.parse_args()
