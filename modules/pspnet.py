import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        #p = F.upsample(input=x, size=(h, w), mode='bilinear')
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


resnet_preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PSPNet(nn.Module):
    def __init__(self, 
                 n_classes=18, 
                 sizes=(1, 2, 3, 6), 
                 psp_size=2048, 
                 deep_features_size=1024, 
                 backend='resnet34',
                 pretrained=True):
        super().__init__()
        #self.feats = getattr(resnets, backend)(pretrained)
        backend_network = torch.hub.load(
            'pytorch/vision:v0.10.0', backend, pretrained=pretrained)
        self.feats = torch.nn.Sequential(*(list(backend_network.children())[:-2]))
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        
        self.up_1 = PSPUpsample(1024, 512)
        self.up_2 = PSPUpsample(512, 256)
        self.up_3 = PSPUpsample(256, 128)
        self.up_4 = PSPUpsample(128, 64)
        self.up_5 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, compute_loss=False):
        x = resnet_preprocess(x)
        #f, class_f = self.feats(x) 
        f = self.feats(x) 
        p = self.psp(f)
        #if compute_loss:
        #    p = self.drop_1(p)
        p = self.up_1(p)
        #if compute_loss:
        #    p = self.drop_2(p)
        p = self.up_2(p)
        #if compute_loss:
        #    p = self.drop_2(p)
        p = self.up_3(p)
        #if compute_loss:
        #    p = self.drop_2(p)
        p = self.up_4(p)
        p = self.up_5(p)
        return self.final(p)
        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        #return self.final(p), self.classifier(auxiliary)


_models = {
    'squeezenet': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', n_classes=n),
    'densenet': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', n_classes=n),
    'resnet18': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', n_classes=n),
    'resnet34': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34', n_classes=n),
    'resnet50': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50', n_classes=n),
    'resnet101': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda n: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152', n_classes=n)
}


def build_network(backend, n_classes):
    backend = backend.lower()
    net = _models[backend](n_classes)
    #net = nn.DataParallel(net)
    return net