import torch.nn as nn
import torch,math,sys
import torch.utils.model_zoo as model_zoo
import functools
from functools import partial

__all__ = ['XResNet', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152',
           'xresnet18_deep', 'xresnet34_deep', 'xresnet50_deep', 'xresnet18_c']

class PrePostInitMeta(type):
    "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        old_init = x.__init__
        def _pass(self): pass
        @functools.wraps(old_init)
        def _init(self,*args,**kwargs):
            self.__pre_init__()
            old_init(self, *args,**kwargs)
            self.__post_init__()
        x.__init__ = _init
        if not hasattr(x,'__pre_init__'):  x.__pre_init__  = _pass
        if not hasattr(x,'__post_init__'): x.__post_init__ = _pass
        return x

class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self): super().__init__()
    def __init__(self): pass

# or: ELU+init (a=0.54; gain=1.55)
act_fn = nn.ReLU(inplace=True)

class Flatten(Module):
    def forward(self, x): 
        return x.view(x.size(0), -1)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True, IN=False):
    if IN == 'IN':
        insNorm = nn.InstanceNorm2d(nf)
        layers = [conv(ni, nf, ks, stride=stride), insNorm]
    else:
        bn = nn.BatchNorm2d(nf)
        nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
        layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(Module):
    def __init__(self, expansion, ni, nh, stride=1, IN=False):
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride, IN=IN),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False, IN=IN)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1, IN=IN),
                   conv_layer(nh, nh, 3, stride=stride, IN=IN),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False, IN=IN)
        ]
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False, IN=IN)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))

def filt_sz(recep): return min(64, 2**math.floor(math.log2(recep*0.75)))

class XResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, c_out=1000, IN=False):
        stem = []
        sizes = [c_in,32,32,64]
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i+1], stride=2 if i==0 else 1, IN=IN))
            #nf = filt_sz(c_in*9)
            #stem.append(conv_layer(c_in, nf, stride=2 if i==1 else 1))
            #c_in = nf

        block_szs = [64//expansion,64,128,256,512] +[256]*(len(layers)-4)
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1 if i==0 else 2, IN=IN)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(block_szs[-1]*expansion, c_out),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride, IN=False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1, IN=IN)
              for i in range(blocks)])

def xresnet(pretrained, expansion, n_layers, name, c_out=1000, **kwargs): #(!) had to move pretrained to first arg, without default
    model = XResNet(expansion, n_layers, c_out=c_out, **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls[name]))
    return model

me = sys.modules[__name__]
for n,e,l in [
    [ 18 , 1, [2,2,2 ,2] ],
    [ 34 , 1, [3,4,6 ,3] ],
    [ 50 , 4, [3,4,6 ,3] ],
    [ 101, 4, [3,4,23,3] ],
    [ 152, 4, [3,8,36,3] ],
]:
    name = f'xresnet{n}'
    setattr(me, name, partial(xresnet, expansion=e, n_layers=l, name=name))

xresnet18_deep = partial(xresnet, expansion=1, n_layers=[2, 2,  2, 2,1,1], name='xresnet18_deep')
xresnet34_deep = partial(xresnet, expansion=1, n_layers=[3, 4,  6, 3,1,1], name='xresnet34_deep')
xresnet50_deep = partial(xresnet, expansion=4, n_layers=[3, 4,  6, 3,1,1], name='xresnet50_deep')

xresnet18_c = partial(xresnet, expansion=1, n_layers=[2, 2, 2 ,2], name='xresnet18_c') #(!)
