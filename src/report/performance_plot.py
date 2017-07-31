import matplotlib.pyplot as plt


class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def draw_performance_epoch(self, performances, epochs, colors, names):
        legend_handles = []
        for p, e, c, n in zip(performances, epochs, colors, names):
            line, _ = plt.plot(range(e), p, c,
                         range(e), p, c+'o',
                         label=n
                  )
            legend_handles.append(line)

        plt.title("Performance of " + self.name + " over the epochs")
        plt.ylim(ymax=1)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(handles=legend_handles)
        plt.show()
