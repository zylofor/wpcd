import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F



class Logger(object):
    def __init__(self, filename="train.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  

    def flush(self):
        pass

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal(m.weight,nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            # ZeroPadding((max_height, max_width)),
            # transforms.RandomHorizontalFlip(p=0.15),
            # transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

        ])

        self.img_list = []
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.img_list.append(os.path.join(root, file))

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(

            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        CBAMLayer(in_features),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        CBAMLayer(in_features),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class WCAE(nn.Module):
    def __init__(self, n_residual_blocks = 9):
        super(CycleGAN_Generate, self).__init__()
        # Encoder
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            CBAMLayer(64),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            encoder += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                CBAMLayer(out_features),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            encoder += [ResidualBlock(in_features)]
        # N 256 56 56

        # Decoder
        out_features = in_features // 2
        decoder = []
        for _ in range(2):
            decoder += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        # N 64 224 224

        # Output layer
        decoder += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()]
        # N 3 224 224

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encode(x)
        return x, self.decode(x)

    def load_pretrained_weights(self, pretrain_path):
        model_dict = self.state_dict()
        self.apply(init_weight)
        pretrained_dict = torch.load(pretrain_path)
        new_model_dict = {}

        for k,v in pretrained_dict.items():
            if k.replace("model","encode") in model_dict:
                new_model_dict.update({k.replace("model","encode"): v})

            if k.replace("model", "decode") == "decode.4.weight":
                continue

            if k.replace("model", "decode") == "decode.4.bias":
                continue

            if k.replace("model", "decode") in model_dict:
                new_model_dict.update({k.replace("model", "decode"): v})

        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.replace("model") in model_dict}
        model_dict.update(new_model_dict)
        self.load_state_dict(model_dict)
        # print(f"Loading pretrained weights for {pretrained_dict}")

# def load_model(model,model_path):
#     # model = CycleGAN_Generate()
#     model.load_state_dict(torch.load(model_path))
#     print(f"Loading the weight file --> {model_path}")
#     return model


def huber_loss(y_true, y_pred, delta=1.0):

    diff = torch.abs(y_true - y_pred)
    delta_tensor = torch.tensor(delta, dtype=torch.float32, device=diff.device)
    quadratic = torch.min(diff, delta_tensor)
    linear = diff - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta_tensor * linear)

def plot_feature(x,epoch,path):
    fig = plt.figure(figsize=(8,8))
    rand_indices = torch.randperm(x.shape[1])[:20]
    for i in  range(20):
        ax = fig.add_subplot(5,4,i+1)
        ax.imshow(x[0,rand_indices[i]].cpu())
        ax.set_xticks([])
        ax.set_yticks([])
    # plt.show()
    plt.savefig(os.path.join(path,f"epoch{epoch}.jpg"),dpi=600)

def train():
    root =  "TSWPCNet/WCAE"
    preTrain_path = "TSWPCNet/WCAE/pretraining.pth"
    embedding_save_path = "output_embedding"
    output_save_path = "output_imgs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataset = ImageDataset(train_image_dir)
    val_dataset = ImageDataset(val_image_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=False)

    model = WCAE()
    # model.load_pretrained_weights(pretrain_path=preTrain_path)
    # model = load_model(model,preTrain_path)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00125, momentum=0.934)

    n_epochs = 1000
    losses = []
    best_loss = float("inf") 
    best_val_loss = float("inf")

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        total_time = 0

        for i, data in tqdm(enumerate(train_dataloader)):
            img = data.to(device)
            encoder_out, decoder_out = model(img)
            loss = huber_loss(decoder_out,img)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            batch_time = time.time() - start_time - total_time
            total_time += batch_time


            remaining_time = total_time * (len(train_dataloader) * (n_epochs - epoch - 1) + len(train_dataloader) - i - 1) / (
                        (i + 1) * len(train_dataloader))
            remaining_time = int(remaining_time)
            remaining_hour = remaining_time // 3600
            remaining_min = (remaining_time % 3600) // 60
            remaining_sec = remaining_time % 60

            if i % 10 == 0:
                print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Time Remaining: {}h {}m {}s, Total Running Time: {}"
                      .format(epoch, n_epochs, i, len(train_dataloader), loss.item(),
                              remaining_hour, remaining_min, remaining_sec,total_time))

                if i % 950 == 0:
                    model.eval()
                    with torch.no_grad():
                        input_imgs = img[:20].permute(0, 2, 3, 1).cpu().detach().numpy()
                        output_imgs = decoder_out[:20].permute(0, 2, 3, 1).cpu().detach().numpy()
                        enc_features = encoder_out[:20].permute(0, 2, 3, 1).cpu().detach().numpy()

                        fig, axs = plt.subplots(nrows=5, ncols=12, figsize=(24, 10))

                        for i in range(20):
                            # input images
                            axs[i // 4, (i % 4) * 3].imshow(input_imgs[i])
                            axs[i // 4, (i % 4) * 3].set_xticks([])
                            axs[i // 4, (i % 4) * 3].set_yticks([])
                            axs[i // 4, (i % 4) * 3].set_title("Input {}".format(i + 1),fontsize=16)

                            # output images
                            axs[i // 4, (i % 4) * 3 + 2].imshow(output_imgs[i])
                            axs[i // 4, (i % 4) * 3 + 2].set_xticks([])
                            axs[i // 4, (i % 4) * 3 + 2].set_yticks([])
                            axs[i // 4, (i % 4) * 3 + 2].set_title("Output {}".format(i + 1),fontsize=16)

                            # encoder features
                            axs[i // 4, (i % 4) * 3 + 1].imshow(enc_features[i, :, :, 0])
                            axs[i // 4, (i % 4) * 3 + 1].set_xticks([])
                            axs[i // 4, (i % 4) * 3 + 1].set_yticks([])
                            axs[i // 4, (i % 4) * 3 + 1].set_title("Embedding {}".format(i + 1),fontsize=16)

                        plt.tight_layout()
                        plt.savefig(os.path.join(root,output_save_path,f"epoch_{epoch}.jpg"),dpi=600)
                        # plt.show()

                        features = model.encode(img)
                        plot_feature(features,epoch,os.path.join(root,embedding_save_path))


        epoch_loss /= len(train_dataloader)

        print(f"Epoch: {epoch},The average loss is {epoch_loss}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            mse_psnr,huber_psnr = 0.0,0.0
            for i, val_data in enumerate(val_dataloader):
                val_img = val_data.to(device)
                val_encoder_out, val_decoder_out = model(val_img)
                val_huber_loss = huber_loss(val_decoder_out, val_img)
                val_loss += val_huber_loss

                # Calulate PSNR
                mse = F.mse_loss(val_decoder_out, val_img)
                max_pixes = torch.max(val_img)
                mse_psnr += (20 * torch.log10(max_pixes / torch.sqrt(mse)))
                huber_psnr += (20 * torch.log10(max_pixes / torch.sqrt(val_huber_loss)))


        val_loss /= len(val_dataloader)
        mse_psnr /= len(val_dataloader)
        huber_psnr /= len(val_dataloader)
        print(f"Epoch: {epoch}, Validation loss: {val_loss}, MSE_PSNR:{mse_psnr}, Huber_PSNR：{huber_psnr}")

        # Save the weights of the model with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print(f"The best validation loss is {best_val_loss} in Epoch {epoch}")
            torch.save(model.state_dict(),
                       f"output_weight/best_wcae_{epoch}.pth")

        model.train()
        # scheduler.step()

    print(f"Training finished! The best validation loss is {best_val_loss}, and best_epoch is {best_epoch}.")



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger()

    train_image_dir = "../WPCD/train_image_dir"
    val_image_dir = "../WPCD/val_image_dir"

    train()
