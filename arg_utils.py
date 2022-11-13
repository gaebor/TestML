from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=256, type=int, help=' ')
    parser.add_argument('--lr', default=0.01, type=float, help=' ')
    parser.add_argument('--device', default='cpu', type=str, help=' ')
    parser.add_argument('--epoch', default=2, type=int, help=' ')
    parser.add_argument(
        '--optimizer', default='Adam', type=str, choices={'Adam', 'NAdam', 'SGD'}, help=' '
    )
    return parser.parse_args()
