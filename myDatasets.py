
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import json
import pandas as pd
import numpy as np
from PIL import Image

class MyDatasets(Dataset):
    def __init__(self, imgDirectory, transform = None):
        self.imgDirectory = imgDirectory
        self.df = self.createDataFrame()
        self.image_ids = self.df['image_id'].unique()
        self.transform = transform

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = Image.open(os.path.join(self.imgDirectory, image_id + '.jpg'))

        if self.transform:
            img = self.transform(img, key = 'train')

        # dataframe
        df = self.df[self.df['image_id'] == image_id]
        boxes = torch.tensor(df[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype = torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype = torch.float32)
        labels = torch.tensor(df['class'].values.astype(np.int64), dtype = torch.int64)
        iscrowd = torch.zeros((df.shape[0], ), dtype = torch.int64)
        
        # target準備
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        return img, target

    def createDataFrame(self):
        labelClass = {'bippa': 1}
        all_files = os.listdir(self.imgDirectory)
        abs_files = [os.path.join(self.imgDirectory, f) for f in all_files]
        jsonPaths = [f for f in abs_files if os.path.splitext(f)[1] == '.json']
        df = pd.DataFrame(columns = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
        for jsonPath in jsonPaths:
            image_id = os.path.splitext(os.path.basename(jsonPath))[0]
            fileName = open(jsonPath, 'r', encoding = 'utf-8')
            jsonData = json.load(fileName)
            points = jsonData['shapes']
            for i in range(len(points)):
                rect = np.array(points[i]['points']).flatten()
                
                df_c = pd.DataFrame([rect], columns = ['xmin', 'ymin', 'xmax', 'ymax'])
                df_c['image_id'] = image_id
                df_c['class'] = labelClass['bippa']
                df = pd.concat([df, df_c])
        
        return df
