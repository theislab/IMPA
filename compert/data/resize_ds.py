"""
Input a dataset of images and resize it to a dimension of interest 
"""

from ..utils import resize_images
import argparse

# Parser 
parser = argparse.ArgumentParser(description='Resize the data')

parser.add_argument('--data_dir', metavar='Data directory', type=str, help='The directory where the data to resize is stored', required = True)
parser.add_argument('--out_dir', metavar='Destination directory', type=str, help='The directory where the resized images are stored', required = True)
parser.add_argument('--width', metavar='Width', type=int, help='New width of the images', default = 64)
parser.add_argument('--height', metavar='Height', type=int, help='New height of the images', default = 64)
parser.add_argument('--interp', metavar='Interpolation', default='cubic',  choices=['nn', 'linear', 'area', 'cubic', 'lanczos'], 
                    help='Type of interpolation')

if __name__ == '__main__':
    args = parser.parse_args()
    resize_images(args.data_dir, args.out_dir, args.width, args.height, args.interp)
