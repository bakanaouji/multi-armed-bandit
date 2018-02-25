import os
import matplotlib.pyplot as plt
import pandas as pd


def summarize_log(save_path, file_name, y_lim):
    files = os.listdir(save_path)
    files = [file for file in files if file[-4:] != '.pdf']

    # summarize data
    df = [pd.DataFrame.from_csv(save_path + '/' + file
                                + '/mean_' + file_name + '.csv')
          for file in files]
    df_concat = pd.DataFrame()
    for data in df:
        df_concat = pd.concat([df_concat, data], axis=1)

    # plot data
    plt.plot(df_concat)
    plt.grid()
    plt.xscale('log')
    plt.xlim(0, len(df_concat.index))
    plt.ylim(0, y_lim)
    plt.savefig(save_path + '/summarize_mean_' + file_name + '.pdf')
    plt.close()
