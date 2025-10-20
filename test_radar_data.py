
# 把雷达数据转化成图片用vit训练分类的测试代码


import argparse
import copy
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim

# === 新增：引入你自己的 ViT ===
from vision_transformer.VisionTransformer import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify radar data converted to images')
    parser.add_argument('--model', type=str, default='vit', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) / len(output)  # 保持你原来的写法
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, criterion, epoch=None, checkpoint=None, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # 对 Validation：记录最优 state_dict（最小改动修正“引用同一对象”的问题）
    if set == "Validation" and checkpoint is not None:
        if test_loss < checkpoint['best_loss']:
            print("new best model: val_loss {}, epoch {}".format(test_loss, epoch))
            checkpoint['best_loss'] = test_loss
            checkpoint['best_epoch'] = epoch
            checkpoint['best_state'] = copy.deepcopy(model.state_dict())


def run(args):
    # 数据增强/归一化：训练与测试分离；使用 CIFAR10 统计
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    # TODO: adjust folder
    dataset = datasets.CIFAR10('./dataset/CIFAR10_train-val', download=True, train=True, transform=train_tf)
    train_len = int(len(dataset) * 0.9)
    trainset, valset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # TODO: adjust folder
    testset = datasets.CIFAR10('./dataset/CIFAR10_test', download=True, train=False, transform=test_tf)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # 构建模型
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)
        # === 关键修正：CIFAR10 是 10 类 ===
        model.fc = nn.Linear(model.fc.in_features, 10)

    elif args.model == "vit":
        # === 用你的 VisionTransformer，参数给一套适合 32x32 CIFAR10 的轻量配置 ===
        model = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_classes=10,
            in_c=3,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            # 下列可选参数若你的实现支持就会生效；不支持也没关系（Python 默认会忽略未知实参需你删掉）
            # qkv_bias=True,
            # representation_size=None,
            # distilled=False,
            # drop_ratio=0.1,
            # attn_drop_ratio=0.0,
            # drop_path_ratio=0.0,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    best_checkpoint = {
        "best_loss": float('INF'),
        "best_epoch": 0,
        "best_state": None
    }

    # 损失函数/设备/优化器（保持你的原设置）
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # 替换原来的 SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    # 训练-验证
    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        test(model, device, valloader, criterion, epoch=epoch, checkpoint=best_checkpoint, set="Validation")

    # 用最优权重在测试集上评估（若未产生 best，就用最终权重）
    if best_checkpoint["best_state"] is not None:
        model.load_state_dict(best_checkpoint["best_state"])
    test(model, device, testloader, criterion)

    # 保存
    if args.save_model:
        torch.save({
            "model_name": args.model,
            "best_epoch": best_checkpoint["best_epoch"],
            "best_loss": best_checkpoint["best_loss"],
            "state_dict": model.state_dict()
        }, "./best_{model_name}_e{epoch}_score{val_loss}.pt".format(
            model_name=args.model,
            epoch=best_checkpoint['best_epoch'],
            val_loss=best_checkpoint['best_loss'])
        )


if __name__ == '__main__':
    args = parse_args()
    run(args)


















