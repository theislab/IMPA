import torch
import torchvision.transforms as T
import yaml
from sklearn.metrics import accuracy_score

def classifier_score(model, data, label):
    y_hat = []
    with torch.no_grad():
        for batch in data:
            pred = model(batch[0], None).argmax(1).to('cpu').tolist()
            y_hat += pred
    y = label*torch.ones(len(y_hat))
    return accuracy_score(y_hat, y)
             
def parse_yaml(path):
    with open(path, "r") as stream:
        raw_config = yaml.safe_load(stream)
    # Check if fixed is present
    if "fixed" in raw_config:
        raw_config=raw_config["fixed"]
    if len(raw_config)==1:
        raw_config = raw_config[list(raw_config.keys())[0]]
        return raw_config
    # Parse
    config = {}
    for key in raw_config:
        parsed_key = key.split('.')[2]
        config[parsed_key] = raw_config[key]
    return config

class Args(dict):
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self

def t2i(tensor):
    tensor = tensor.detach().cpu().permute(1,2,0).numpy()
    return (tensor+1.)/2.

class CustomTransform:
    """
    Scale and resize an input image 
    """
    def __init__(self, augment=False, normalize=False, dim=0):
        self.augment = augment 
        self.normalize = normalize 
        self.dim = dim
        
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
            t.append(T.Normalize(mean=[0.5 for _ in range(X.shape[self.dim])], std=[0.5 for _ in range(X.shape[self.dim])]))
        
        # Perform augmentation step
        if self.augment:
            t.append(T.RandomHorizontalFlip(p=0.3))
            t.append(T.RandomVerticalFlip(p=0.3))

        trans = T.Compose(t)
        return trans(X)
    