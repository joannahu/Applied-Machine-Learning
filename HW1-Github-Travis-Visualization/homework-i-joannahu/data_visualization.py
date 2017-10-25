import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def scatterplot_matrix():
    """Plots 4x4=16 number of subplots about sepal length and petal width of iris.
    The diagonals contain histograms, the different species  distinguished by color and there's a legend for the species.
    """

    # load data
    iris_dataset = load_iris()
    data = iris_dataset
    setosa = data['data'][data['target'] == 0]
    versicolor = data['data'][data['target'] == 1]
    virginica = data['data'][data['target'] == 2]

    # set picture frame
    num = 4
    fig, axes = plt.subplots(nrows=num, ncols=num, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)

    # set scatter plot
    for i in range(0, num):
        for j in range(0, num):
            if i == j:
                continue
            axes[j, i].plot(setosa[:, j], setosa[:, i], color='navy', marker='o', linestyle='none')
            axes[j, i].plot(versicolor[:, j], versicolor[:, i], color='purple', marker='*', linestyle='none')
            axes[j, i].plot(virginica[:, j], virginica[:, i], color='pink', marker='s', linestyle='none')

    # set histgram on the diagram
    for i in range(0, num):
        axes[i, i].hist(setosa[:, i], color='navy')
        axes[i, i].hist(versicolor[:, i], color='purple')
        axes[i, i].hist(virginica[:, i], color='pink')

    axes[0, 0].set_title('Sepal length')
    axes[1, 1].set_title('Sepal width')
    axes[2, 2].set_title('Petal length')
    axes[3, 3].set_title('Petal width')

    plt.legend(('Setosa', 'Virginica', 'Versicolor'))  # add legend

    # add Main title
    fig.suptitle('Iris Plots, measurements in cm', size=20)
    plt.show()


scatterplot_matrix()
