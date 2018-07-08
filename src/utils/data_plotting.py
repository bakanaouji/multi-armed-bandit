import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_each_data(folder_name, file_name, y_min, y_max, data=None, legend=None):
    if data is None:
        data = pd.DataFrame.from_csv(folder_name + '/' + file_name + '.csv')
    plt.plot(data)
    plt.grid()
    plt.xscale('log')
    plt.xlim(0, len(data.index))
    plt.ylim(y_min, y_max)
    if legend is not None:
        plt.legend(legend, loc='upper left')
    plt.savefig(folder_name + '/' + file_name + '.pdf')
    plt.close()


def plot_summarized_data(folder_name, file_name, y_min, y_max):
    files = os.listdir(folder_name)
    files = [file for file in files if file[-4:] != '.pdf' and file[:1] != '.']

    # summarize data
    df = [pd.DataFrame.from_csv(folder_name + '/' + file
                                + '/mean_' + file_name + '.csv')
          for file in files]
    df_concat = pd.DataFrame()
    for data in df:
        df_concat = pd.concat([df_concat, data], axis=1)

    # plot data
    plot_each_data(folder_name, 'summarized_maen_' + file_name, y_min, y_max,
                   df_concat, files)
