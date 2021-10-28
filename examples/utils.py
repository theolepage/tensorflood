import numpy as np
import pathlib
import glob
import imageio
import matplotlib.pyplot as plt

from tensorflood.engine import Graph, Variable

def save_contour_figure(X, y, model, epoch, stride=0.1):
    # Generate a grid of points to draw a contour
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, stride),
        np.arange(y_min, y_max, stride)
    )
    
    # Predict class of grid points
    grid = np.concatenate((xx.reshape((-1, 1)), yy.reshape((-1, 1))), axis=-1)
    Z = model(Variable(grid))
    preds = np.where(Z.data < 0.5, 0, 1).reshape(xx.shape)

    # Plot contour and training data
    plt.figure(figsize=(7, 7))
    plt.contourf(xx, yy, preds, cmap=plt.cm.Spectral, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('off')
    plt.box(on=None)
    label = 'Epoch {0}'.format(epoch)
    plt.text(-1.6, 1.8, label, ha='center', va='center', size='large')
    
    # Save figure
    pathlib.Path("tmp").mkdir(exist_ok=True)
    plt.savefig("tmp/{}".format(epoch), transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def create_contour_gif(path):
    filenames = glob.glob('tmp/*.png')
    filenames = sorted(filenames, key=lambda x: int(x.split('/')[-1][:-4]))
    images = [imageio.imread(f) for f in filenames]
    imageio.mimsave(path, images, fps=10)