import torchvision.transforms as T
import torch

class CustomTransform:
    """
    Scale and resize an input image 
    """
    def __init__(self, augment=False, normalize=False):
        self.augment = augment 
        self.normalize = normalize 
        
    def __call__(self, X):
        """
        By random horizontal and vertical flip
        """
        # Add random noise and rescale pixels between 0 and 1 
        random_noise = torch.rand_like(X)  # To transform the image to continuous data point
        X = (X+random_noise)/255.0  # Scale to 0-1 range
        
        t = []
        # Normalize the input between -1 and 1 
        if self.normalize==True:
            t.append(T.Normalize(mean=[0.5 for _ in range(X.shape[0])], std=[0.5 for _ in range(X.shape[0])]))
        
        # Perform augmentation step
        if self.augment:
            t.append(T.RandomHorizontalFlip(p=0.3))
            t.append(T.RandomVerticalFlip(p=0.3))

        trans = T.Compose(t)
        return trans(X)
        