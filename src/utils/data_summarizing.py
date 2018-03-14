import os
import pandas as pd

from utils.data_plotting import plot_data


def summarize_data(save_path, file_name, y_lim):
    files = os.listdir(save_path)
    files = [file for file in files if file[-4:] != '.pdf' and file[:1] != '.']

    # summarize data
    df = [pd.DataFrame.from_csv(save_path + '/' + file
                                + '/mean_' + file_name + '.csv')
          for file in files]
    df_concat = pd.DataFrame()
    for data in df:
        df_concat = pd.concat([df_concat, data], axis=1)

    # plot data
    plot_data(save_path, 'summarize_maen_' + file_name, 0.0, y_lim, df_concat)
