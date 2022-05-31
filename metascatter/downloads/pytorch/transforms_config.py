import torchvision.transforms as transforms

finalsize = 224

data_transforms = {
    'inference': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(finalsize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}