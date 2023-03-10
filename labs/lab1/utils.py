from matplotlib import pyplot as plt

def visualize_2D(x,y,x_title='',y_title='',title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set( title=title,
           ylabel=y_title, xlabel=x_title)
    plt.plot(x,y)
    plt.show()
