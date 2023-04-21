from matplotlib import pyplot as plt



def visualize_2D(x,y,x_title='',y_title='',title=None):
    plt.ion()  # 打开交互模式
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set( title=title,
           ylabel=y_title, xlabel=x_title)
    plt.plot(x,y)
    
    plt.pause(1)
    plt.ioff()  # 关闭交互模式
    plt.show()



       


