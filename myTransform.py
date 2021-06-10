from torchvision import transforms

class MyTransform():
    def __init__(self):
        pass

    def __call__(self, img, key):
        self.transform = {
            'train': transforms.Compose([
                # transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(degrees = 45),
                transforms.ToTensor(),
                # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),

            ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ])
        }
        return self.transform[key](img)
