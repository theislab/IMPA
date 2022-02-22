import matplotlib.pyplot as plt 
import os

class Plotter:
    def __init__(self, dest_dir):
        self.dest_dir = dest_dir

    def plot_reconstruction(self, original, reconstruction, epoch, save=True, plot=False):
        """
        Plot the original and reconstructed channels one over the other 
        """
        fig = plt.figure(constrained_layout=True, figsize = (10,10))
        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=2, ncols=1)
        titles = ['ORIGINAL', 'RECONSTRUCTED']
        images = [original, reconstruction]        

        for row, subfig in enumerate(subfigs):
            subfig.suptitle(titles[row], fontsize = 15)

            # create 1x3 subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=5)
            for col, ax in enumerate(axs):
                ax.imshow(images[row][:,:,col], cmap = 'Greys')
                ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.dest_dir, 'reconstructions', f'reconstructions_epoch_{epoch}.png'))
        if plot:
            plt.plot()
    
    def plot_channel_panel(self, image, epoch, save, plot):
        """
        Plot a panel of single channel figures 
        -------------------
        image: numpy array of dimension (height, width, 5)
        """
        fig, axs = plt.subplots(1, 5, figsize = (10,10))
        for z in range(image.shape[-1]):
            axs[z].imshow(image[:,:,z], cmap = 'Greys')
            axs[z].axis('off')
        plt.title('Generated sample', fontsize=15)
        if save:
            plt.savefig(os.path.join(self.dest_dir, 'generations', f'generations_epoch_{epoch}.png'))
        if plot:
            plt.plot()
