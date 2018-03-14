import argparse

import os

from utils.data_plotting import plot_data


def main():
    parser = argparse.ArgumentParser(description='Summarize data')

    # setting of experiment
    parser.add_argument('--y_min', type=float, default=-2.0,
                        help='Min value of y axis')
    parser.add_argument('--y_max', type=float, default=2.0,
                        help='Max value of y axis')
    parser.add_argument('--save_path',
                        default='N(1.0,9.0)N(0.0,0.09)'
                                '/ThompsonSamplingGaussianPrior'
                        , help='Path to save log')
    parser.add_argument('--file_name', default='theta',
                        help='File name to be summarized')

    args = parser.parse_args()

    save_path = '../data/' + args.save_path

    # summarize data
    print('----------Plot Data----------')
    files = os.listdir(save_path)
    files = [file for file in files
             if file[-4:] != '.csv' and files[-4:] != '.pdf'
             and file[:1] != '.']
    for file in files:
        print('Plot Data : ' + save_path + '/' + file + '/'
              + args.file_name + '.csv')
        plot_data(save_path + '/' + file, args.file_name,
                  args.y_min, args.y_max)


if __name__ == '__main__':
    main()
