import torch.utils.data
import tarfile
import os
import urllib.request
import pickle



class MNISTStroke(torch.utils.data.Dataset):
    """
    From https://github.com/galatolofederico/pytorch-mnist-stroke/tree/master
    """
    url = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
    
    def __init__(self, loc, train=True, transform=None):
        self.location = loc if loc[-1] == "/" else loc+"/"
        self.transform = transform
        if not self.checkDownload():
            self.download()
        if not self.checkProcessed():
            self.process()
        
        if train:
            self.data = pickle.load(open(self.location+"processed/train", "rb"))
        else:
            self.data = pickle.load(open(self.location+"processed/test", "rb"))
        
    def __getitem__(self, index):
        input = self.data[index]["input"]
        label = self.data[index]["label"]

        if self.transform is not None:
            input = self.transform(input)
        
        return input, label

    def __len__(self):
        return len(self.data)

    def checkDownload(self):
        return os.path.isdir(self.location+"raw")
    
    def checkProcessed(self):
        return os.path.isdir(self.location+"processed")

    def download(self):
        print("Downloading dataset")
        if not os.path.isdir(self.location): os.mkdir(self.location)
        os.mkdir(self.location+"raw") 
        urllib.request.urlretrieve(MNISTStroke.url, self.location+"raw/seq.tar.gz")
        print("Extracting dataset (it might take long time)")
        tar = tarfile.open(self.location+"raw/seq.tar.gz", "r:gz")
        tar.extractall(self.location+"raw/")
        tar.close()
    
    def processFile(self, file):
        f = open(file)
        return torch.tensor([[float(c) for c in line.rstrip().split(" ")] for line in f.readlines()])

    def zerofill(self, input, max_len):
        z = torch.zeros(max_len, input.shape[1])
        z[0:input.shape[0]] = input
        return z

    def process(self):
        print("Processing dataset (it might take long time)")
        test_data = {}
        train_data = {}
        max_len = 0
        for f in os.listdir(self.location+"raw/sequences"):
            parts = f.split("-")
            if len(parts) > 1 and parts[2] == "inputdata.txt":
                data = {
                    "input": self.processFile(self.location+"raw/sequences/"+f)
                }
                if data["input"].shape[0] > max_len: max_len = data["input"].shape[0]  
                if parts[0] == "trainimg":
                    train_data[int(parts[1])] = data
                if parts[0] == "testimg":
                    test_data[int(parts[1])] = data
        
        f = open(self.location+"raw/sequences/trainlabels.txt")
        for i, label in enumerate(f):
            train_data[i]["label"] = torch.tensor(int(label))

        f = open(self.location+"raw/sequences/testlabels.txt")
        for i, label in enumerate(f):
            test_data[i]["label"] = torch.tensor(int(label))
        
        for i, data in test_data.items():
            test_data[i]["input"] = self.zerofill(data["input"], max_len)

        for i, data in train_data.items():
            train_data[i]["input"] = self.zerofill(data["input"], max_len)
        
        os.mkdir(self.location+"processed")
        pickle.dump(test_data, open(self.location+"processed/test", "wb"))
        pickle.dump(train_data, open(self.location+"processed/train", "wb"))