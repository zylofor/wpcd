import time

import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
# from cgan_Pretrain_model import CycleGAN_Generate
from WAE import CycleGAN_Generate,load_model
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap


def collate_custom(batch):
    """ Custom collate function """
    import numpy as np
    import collections
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))

# 定义一个自定义数据集类，用于加载图像和标签
class ImageFolderWithLabel(Dataset):
    def __init__(self, root):
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            for image_path in os.listdir(label_path):
                image_path = os.path.join(label_path, image_path)
                self.image_paths.append(image_path)
                self.labels.append(int(label))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)


def load_data(img_path_train,img_path_test):
    train_dataset = ImageFolderWithLabel(img_path_train)
    test_dataset = ImageFolderWithLabel(img_path_test)
    train_dataloader = DataLoader(train_dataset,shuffle=True,collate_fn=collate_custom)
    test_dataloader = DataLoader(test_dataset,shuffle=False,collate_fn=collate_custom)
    return train_dataloader,test_dataloader

def PCA_down(train_codes, out_size):
    pca = PCA(out_size)
    train_codes = pca.fit_transform(train_codes)
    # U,S,V = torch.pca_lowrank(train_codes,q=out_size)
    return train_codes
    # return torch.matmul(train_codes,V)

def extract_features(model,train_loader,test_loader,out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device=device)
    train_codes,train_labels = [],[]
    test_codes,test_labels = [],[]

    with torch.no_grad():
        for i,data in enumerate(tqdm(train_loader)):
            inputs,labels = data[0].to(device),data[1].to(device)
            #Replace
            codes,_ = model(inputs)
            codes = inputs

            train_codes.append(codes.detach().view(codes.shape[0], -1))
            train_labels.append(labels)

    train_codes = torch.cat(train_codes).cpu()
    train_labels = torch.cat(train_labels).cpu()


    print("Starting decomposition for train...")
    # train_codes = PCA_down(train_codes, min(train_codes.shape[0], 64))
    # umap_obj = umap.UMAP(n_neighbors=20, min_dist=0, n_components=5)
    # train_codes = umap_obj.fit_transform(train_codes)
    tsne = TSNE(n_components=2, random_state=0)
    train_codes = tsne.fit_transform(train_codes)

    #画图
    # colors = ['#DC143C', '#FF00FF', '#0000FF', '#00fa9a', '#ffd700', '#ff1493', '#e6e6fa', '#00ffff']
    # print(len(set(train_labels)))
    # print(colors[:len(set(train_labels))])
    # cmap = ListedColormap(colors[:len(set(train_labels))])
    # plt.scatter(train_codes[:,0],train_codes[:,1],c=train_labels,cmap=cmap)
    # plt.legend()
    # plt.xlabel('Component 1',fontsize=15)
    # plt.ylabel('Component 2',fontsize=15)
    # plt.subplots_adjust()
    # plt.title('Epoch 200',fontsize=15)
    # plt.colorbar()
    # plt.show()
    # print(time.time())
    print("Finishing decomposition...")


    save_location = os.path.join(out_path)
    print("Saving train embeddings...")
    print(f"train codes dims = {train_codes.shape}")

    torch.save(train_codes, os.path.join(save_location, "train_data.pt"))
    torch.save(train_labels, os.path.join(save_location, "train_labels.pt"))
    print("Saved train embeddings!")
    del train_codes, train_labels

    # for i, data in enumerate(tqdm(test_loader)):
    #     with torch.no_grad():
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         codes,_ = model(inputs)
    #         codes = torch.nn.functional.adaptive_avg_pool2d(codes,1)
    #         test_codes.append(codes.view(codes.shape[0], -1))
    #         test_labels.append(labels)
    #
    # test_codes = torch.cat(test_codes).cpu()
    # test_labels = torch.cat(test_labels).cpu()
    #
    # print("Starting umap for val...")
    # test_codes = umap_obj.transform(test_codes)
    # print("Finishing umap...")
    #
    # print("Saving test embeddings...")
    # print(f"test codes dims = {test_codes.shape}")
    # torch.save(test_codes, os.path.join(save_location, "test_data.pt"))
    # torch.save(test_labels, os.path.join(save_location, "test_labels.pt"))
    # print("Saved test embeddings!")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_path = "/home/dell/jzy/lab315/jzy/PaperLearning/MyUbuntu/AE_Cluster/VGG_AE_Cluster/CGAN/+cbam/output_weight/best_gan_cbam_1098.pth"
    img_train_path = "/home/dell/jzy/lab315/jzy/PaperLearning/MyUbuntu/DeepDPM/data/Cluster"
    img_test_path = "/home/dell/jzy/lab315/jzy/PaperLearning/MyUbuntu/DeepDPM/data/test"
    embbeding_out_path = "/home/dell/jzy/lab315/jzy/PaperLearning/MyUbuntu/DeepDPM/embbeding_results"

    model = CycleGAN_Generate()
    model = load_model(model, model_path)

    train_loader,test_loader = load_data(img_train_path,img_test_path)
    print(len(train_loader),len(test_loader))
    extract_features(model,train_loader,test_loader,embbeding_out_path)

if __name__ == '__main__':
    main()
