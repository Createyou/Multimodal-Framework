import os
import sys
import json
import pickle
import random
import re, csv, glob
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    flower_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  #
    train_images_label = []  #
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        images.sort()

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:

        plt.bar(range(len(flower_class)), every_class_num, align='center')

        plt.xticks(range(len(flower_class)), flower_class)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')

        plt.ylabel('number of images')

        plt.title('flower class distribution')
        plt.show()
    train_cnt = Counter(train_images_label)
    val_cnt = Counter(val_images_label)
    print("\nPer-class counts:")
    for cls_name in flower_class:
        idx = class_indices[cls_name]
        total = every_class_num[idx]
        t = train_cnt.get(idx, 0)  #
        v = val_cnt.get(idx, 0)  #
        print(f"  [{idx}] {cls_name:12s}  total={total:4d}  train={t:4d}  val={v:4d}")

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):

            img = images[i].numpy().transpose(1, 2, 0)

            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, signals, labels = data

        pred = model(images.to(device),signals.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (images, signals,labels) in enumerate(data_loader):
        preds = model(images.to(device),signals.to(device))
        preds = torch.max(preds, dim=1)[1]  # 取预测类别索引

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return acc,precision, recall, f1


def read_split_three_data(root: str, train_rate: float = 0.7, val_rate: float = 0.1, test_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    train_images_path, train_images_label = [], []
    val_images_path, val_images_label = [], []
    test_images_path, test_images_label = [], []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in classes:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))


        random.shuffle(images)
        total_num = len(images)
        train_end = int(total_num * train_rate)
        val_end = train_end + int(total_num * val_rate)


        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        train_images_path.extend(train_images)
        val_images_path.extend(val_images)
        test_images_path.extend(test_images)

        train_images_label.extend([image_class] * len(train_images))
        val_images_label.extend([image_class] * len(val_images))
        test_images_label.extend([image_class] * len(test_images))

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for testing.".format(len(test_images_path)))

    train_cnt = Counter(train_images_label)
    val_cnt = Counter(val_images_label)
    test_cnt = Counter(test_images_label)
    print("\nPer-class counts:")
    for cls_name in classes:
        idx = class_indices[cls_name]
        total = every_class_num[idx]
        t = train_cnt.get(idx, 0)
        v = val_cnt.get(idx, 0)
        te = test_cnt.get(idx, 0)
        print(f"  [{idx}] {cls_name:12s}  total={total:4d}  train={t:4d}  val={v:4d}  test={te:4d}")

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label


def evaluate_all_checkpoints(weights_dir: str,
                             csv_out: str,
                             test_loader,
                             device,
                             build_model,
                             weights_key: str = ""):

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)


    pat = re.compile(r"model-(\d+)\.pth$")
    paths = []
    for p in glob.glob(os.path.join(weights_dir, "model-*.pth")):
        m = pat.search(os.path.basename(p))
        if m:
            paths.append((int(m.group(1)), p))
    paths.sort(key=lambda x: x[0])

    if not paths:
        print(f"[WARN] No checkpoints found in: {weights_dir}")
        return


    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "acc", "precision", "recall", "f1", "weights_path"])

        for epoch, wpath in paths:

            model = build_model.to(device)
            ckpt = torch.load(wpath, map_location="cpu")

            if isinstance(ckpt, dict) and weights_key and weights_key in ckpt:
                state_dict = ckpt[weights_key]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt


            new_sd = {}
            for k, v in state_dict.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                new_sd[nk] = v

            missing, unexpected = model.load_state_dict(new_sd, strict=False)
            if missing or unexpected:
                print(f"[WARN][epoch {epoch}] missing={len(missing)} unexpected={len(unexpected)}")

            model.eval()
            acc, precision, recall, f1 = evaluate(model=model,
                                                  data_loader=test_loader,
                                                  device=device)

            print(f"[TEST epoch {epoch}] acc={acc:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
            writer.writerow([epoch, f"{acc:.6f}", f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}", wpath])

    print(f"[Saved] per-epoch test metrics -> {csv_out}")

