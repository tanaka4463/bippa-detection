import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import os
import glob
from PIL import ImageDraw

def trainTestDatasetsSplit(datasets, test_size):
    train_index, test_index = train_test_split(range(len(datasets)), test_size = test_size, shuffle = False)
    train_dataset = Subset(datasets, train_index)
    test_dataset = Subset(datasets, test_index)

    return train_dataset, test_dataset

def collate_fn(batch):
    return tuple(zip(*batch))

def dataLoaderSplit(train_dataset, test_dataset, batch_size = 32, shuffle = True):
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=collate_fn)

    return {'train': train_loader, 'test': test_loader}

def img_convert(img):
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy() * 255
    img = img.astype(np.uint8)
    return img

def plot_img(datasets, index):
    img, target = datasets[index]
    img = img_convert(img)
    img = np.ascontiguousarray(img)
    boxes = target['boxes'].numpy()
    for i in boxes:
        cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), thickness = 2)
    plt.imshow(img)
    plt.show()

def createModel(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model 

def train(epochs, model, dataLoaders, optimizer):
    torch.cuda.empty_cache()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loss_min = 0.9
    results = []
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = []
        model.train()
        for i, batch in tqdm.tqdm(enumerate(dataLoaders['train'])):
            imgs, targets = batch
            
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # loss
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss.append(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        train_meanLoss = np.mean(train_loss)
        results.append(train_meanLoss)
        print(f'Epoch train loss is {train_meanLoss}')
        if train_meanLoss <= train_loss_min:
            train_loss_min = train_meanLoss
            best_weight = copy.deepcopy(model.state_dict())

    # modelの保存
    if not os.path.exists('model'):
        os.mkdir('model')
    torch.save(model.state_dict(), 'model/model.pth')

def inference(imgDirectory, index):
    file_path = os.path.dirname(__file__)
    model = createModel(num_classes = 1 + 1)
    model_path = './model/model.pth'
    model.load_state_dict(torch.load(model_path))

    img_path = glob.glob(os.path.join(imgDirectory, '*.jpg'))
    img = cv2.imread(img_path[index])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = img.astype(np.float32) / 255
    img_tensor = torch.from_numpy(img_tensor)
    img_tensor = img_tensor.permute(2, 0, 1)

    model.eval()
    outputs = model([img_tensor])

    boxes = outputs[0]["boxes"].data.cpu().numpy()
    scores = outputs[0]["scores"].data.cpu().numpy()

    boxes = boxes[scores >= 0.7].astype(np.int32)
    for i, box in enumerate(boxes):
        # label = category[labels[i]]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color = (255, 0, 0), thickness = 3)

    plt.imshow(img)
    plt.show()
