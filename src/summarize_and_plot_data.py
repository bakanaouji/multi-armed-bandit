import argparse

from utils.data_summarizing import summarize_data


def main():
    parser = argparse.ArgumentParser(description='Summarize data')

    # setting of experiment
    parser.add_argument('--y_lim', type=int, default=100,
                        help='Max value of y axis')
    parser.add_argument('--save_path', default='N(1.0,9.0)N(0.0,0.09)',
                        help='Path to save log')
    parser.add_argument('--file_name', default='regret',
                        help='File name to be summarized')

    args = parser.parse_args()

    save_path = '../data/' + args.save_path

    # summarize data
    summarize_data(save_path, args.file_name, args.y_lim)


if __name__ == '__main__':
    main()
