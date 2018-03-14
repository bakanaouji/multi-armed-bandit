import os
import pandas as pd


def calc_mean_data(folder_name):
    files = os.listdir(folder_name)
    files = [file for file in files if file[:3] == 'Exp']
    N = len(files)

    file_names = os.listdir(folder_name + '/Exp0')
    file_names = [file_name for file_name in file_names
                  if file_name[-4:] == '.csv']

    for file_name in file_names:
        df = [pd.DataFrame.from_csv(folder_name + '/Exp' + str(i) + '/'
                                    + file_name) for i in range(N)]
        mean = df[0]
        for i in range(1, N):
            mean = mean + df[i]
        mean = mean / N
        mean.to_csv(folder_name + '/mean_' + file_name)
        print("Save" + folder_name + '/mean_' + file_name)
