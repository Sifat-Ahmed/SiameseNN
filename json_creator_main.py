
import argparse
from json_helper.json_creator import JsonCreator

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--train-dir',
                        help='Training image directory',
                        required=True,
                        type=str)

    parser.add_argument('--train-output',
                        help='Output Training Json Name',
                        required=False,
                        type=str,
                        default='dataset_train')

    parser.add_argument('--val-dir',
                        help='Validation image directory',
                        required=False,
                        type=str)

    parser.add_argument('--val-output',
                        help='Output Validation Json Name',
                        required=False,
                        type=str,
                        default='dataset_val')

    parser.add_argument('--test-dir',
                        help='Test Image directory',
                        required=False,
                        type=str,
                        default=None)

    parser.add_argument('--test-output',
                        help='Output Test Json Name',
                        required=False,
                        type=str,
                        default='dataset_test')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("Creating Training Dataset")
    train = JsonCreator(args.train_dir, args.train_output)
    train.make()

    if args.val_dir:
        print("Creating Validation Dataset")
        valid = JsonCreator(args.val_dir, args.val_output)
        valid.make()

    if args.test_dir:
        print("Creating Test Dataset")
        test = JsonCreator(args.test_dir, args.test_output)
        test.make()