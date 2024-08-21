import torch
from torch import nn

from bbox_tools import multibox_target
from data import load_data_bananas
from ssd import TinySSD
from util import Accumulator


cls_loss = nn.CrossEntropyLoss(reduction="none")
bbox_loss = nn.L1Loss(reduction="none")


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = (
        cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1))
        .reshape(batch_size, -1)
        .mean(dim=1)
    )
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def train():
    num_epochs = 20
    net = TinySSD(num_classes=1)
    batch_size = 32
    train_iter, _ = load_data_bananas(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    net = net.to(device)
    epoch_metrics = {}
    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # Calculate the loss function using the predicted and labeled values
            # of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(
                cls_eval(cls_preds, cls_labels),
                cls_labels.numel(),
                bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                bbox_labels.numel(),
            )
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        epoch_metrics[epoch] = dict(cls_err=cls_err, bbox_mae=bbox_mae)
    return epoch_metrics

if __name__ == '__main__':
    metric = train()
    print(metric)
