import matplotlib.pyplot as plt 
import os

"""
Class for plotting the results 
"""

class Plotter:
    def __init__(self, dest_dir):
        self.dest_dir = dest_dir

    def plot_reconstruction(self, original, reconstruction, epoch, save=True, plot=False):
        """Plot two images represented the original and reconstructed outputs of a neural network.

        Args:
            original (np.array): Original image
            reconstruction (np.array): Reconstructed version of the image 
            epoch (int): Computaion epoch at which image was produced  
            save (bool, optional): Whether to save the plot. Defaults to True.
            plot (bool, optional):  Used in notebooks. Controls whetehr the image should be plot. Defaults to False.
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
                ax.imshow(images[row][:,:,col], cmap = 'gray')
                ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.dest_dir, 'reconstructions', f'reconstructions_epoch_{epoch}.png'))
        if plot:
            plt.plot()
    
    def plot_channel_panel(self, image, epoch, save, plot, title='Generated sample', rgb = False, size = 10, dim = 5):
        """Plot a cell painting image either as a single channel or a multiple channel 

        Args:
            image (np.array): An image as a numpy array with dimensions HxWxC
            epoch (int): Computaion epoch at which image was produced  
            save (bool): Whether the image should be saved 
            plot (bool): Used in notebooks. Controls whetehr the image should be plot
            title (str, optional): Title of the plot. Defaults to 'Generated sample'.<
            rgb (bool, optional): If the images should be plotted RGB (True) or in 5 channels (False). Defaults to False.
        """
        if not rgb:
            fig, axs = plt.subplots(1, dim, figsize = (size,4))
            fig.suptitle(title, fontsize=16)
            for z in range(image.shape[-1]):
                axs[z].imshow(image[:,:,z], cmap = 'gray')
                axs[z].axis('off')
        
        else:
            fig = plt.figure(figsize  = (size,size))
            plt.imshow(image[:,:,[0,2,4]])
            plt.axis('off')

        if save:
            plt.savefig(os.path.join(self.dest_dir, 'generations', f'generations_epoch_{epoch}.png'))
        if plot:
            plt.plot()
