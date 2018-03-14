import argparse

import os

from utils.data_plotting import plot_each_data


def main():
    parser = argparse.ArgumentParser(description='Plot each data')

    # setting of experiment
    parser.add_argument('--y_min', type=float, default=0.0,
                        help='Min value of y axis')
    parser.add_argument('--y_max', type=float, default=100.0,
                        help='Max value of y axis')
    parser.add_argument('--folder_name',
                        default='N(1.0,9.0)N(0.0,0.09)'
                                '/ThompsonSamplingGaussianPrior',
                        help='Folder name where logs of experiments are saved')
    parser.add_argument('--file_name', default='regret',
                        help='File name to plot')

    args = parser.parse_args()

    folder_name = '../data/' + args.folder_name

    # summarize data
    print('----------Plot Data----------')
    files = os.listdir(folder_name)
    files = [file for file in files
             if file[-4:] != '.csv' and files[-4:] != '.pdf'
             and file[:1] != '.']
    for file in files:
        print('Plot Data : ' + folder_name + '/' + file + '/'
              + args.file_name + '.csv')
        plot_each_data(folder_name + '/' + file, args.file_name,
                       args.y_min, args.y_max)


if __name__ == '__main__':
    main()
