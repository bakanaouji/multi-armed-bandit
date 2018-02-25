import os
import pandas as pd


def calc_mean_log(save_path, file_name):
    files = os.listdir(save_path)
    files = [file for file in files if file[:3] == 'Exp']
    N = len(files)
    df = [pd.DataFrame.from_csv(save_path + '/Exp' + str(i) + '/'
                                + file_name + '.csv') for i in range(N)]
    mean = pd.DataFrame()
    for i in range(N):
        mean = pd.concat([mean, df[i]], axis=1)
    mean = mean.mean(axis=1)
    mean.to_csv(save_path + '/mean_' + file_name + '.csv')

    print(mean)
