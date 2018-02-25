import matplotlib.pyplot as plt
import pandas as pd


def plot_data(save_path, file_name, y_min, y_max, data=None):
    if data is None:
        data = pd.DataFrame.from_csv(save_path + '/' + file_name + '.csv')
    plt.plot(data)
    plt.grid()
    plt.xscale('log')
    plt.xlim(0, len(data.index))
    plt.ylim(y_min, y_max)
    plt.savefig(save_path + '/' + file_name + '.pdf')
    plt.close()
