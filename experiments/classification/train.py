"""Module to train the ViTransformer for image net classification.
"""
import torch
import torchvision
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from models.classification.vision_transformer import VisionTransformer
from datasets.classification.image_net.dataset import ImageNet

def train_model() -> None:
    """Function to train the model."""
    writer = SummaryWriter()
    dataset = ImageNet(csv_file="/home/ahsan/datasets/image-net/image-net-50.csv",
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.Resize(size=224),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.CenterCrop(size=224), #RandomCrop(size=224),
                           torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                       ]))
    model = VisionTransformer(img_size=224, in_channels=3, patch_size=16, embed_size=16*16*3,
                              num_layers=12, num_heads=8, head_dim=256, mlp_dim=512,
                              num_classes=50, drop_prob=0.0)
    model.cuda()

    optimizer = Adam(params=model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    step = 0
    for _ in range(500):
        print(f'Epoch:{_}')
        for images, labels in dataloader:
            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

            pred, att_weights = model(images)

            loss = criterion(pred, labels)
            loss.backward()

            optimizer.step()

            #inputs = images[:4, :, :, :].cpu()
            #outputs = outputs.detach()[:4, :].cpu()
            for head_idx in range(8):
                writer.add_image(tag=f"head{head_idx}",
                                 img_tensor=(
                                    att_weights[11][0, head_idx, :, :].detach().cpu().unsqueeze(0) *
                                        255).to(torch.uint8),
                                    global_step=step)
            writer.add_image(tag="input", img_tensor=images[0, :, :, :], global_step=step)
            writer.add_scalar("loss", loss.item(), global_step=step)
            step += 1
            #out
            #print(loss.item())

train_model()
    