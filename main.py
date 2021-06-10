from myDatasets import MyDatasets
from myTransform import MyTransform
from utils import *

def main():
    imgDirectory = 'img'
    transform = MyTransform()
    datas = MyDatasets(imgDirectory = imgDirectory, transform = transform)
    train_dataset, test_dataset = trainTestDatasetsSplit(datas, test_size = 0.2)
    dataLoaders = dataLoaderSplit(train_dataset, test_dataset, batch_size = 2, shuffle = True)

    model = createModel(num_classes = 1 + 1) # target + background
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
    
    epochs = 2
    train(epochs = epochs, model = model, dataLoaders = dataLoaders, optimizer = optimizer)

main()
