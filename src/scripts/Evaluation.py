import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix, accuracy_score


def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return (1 + beta ** 2.) * (p * r) / float(beta ** 2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def miou_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def specificity_score(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    else:
        specificity = 0.0
    return specificity

def compute_asd(gt, pred):
        # 计算预测掩码和真实掩码的距离场
        pred_distance = distance_transform_edt(pred)
        gt_distance = distance_transform_edt(gt)

        # 计算平均对称距离
        asd = (np.sum(pred_distance * gt) + np.sum(gt_distance * pred)) / (np.sum(pred) + np.sum(gt))

        return asd

def compute_hd(gt, pred):
        # 选择沿着第一个轴的最大切片
        pred_slice = pred.max(axis=0)
        gt_slice = gt.max(axis=0)

        # 计算Hd
        hd1 = directed_hausdorff(pred_slice, gt_slice)[0]
        hd2 = directed_hausdorff(gt_slice, pred_slice)[0]
        hd = max(hd1, hd2)

        return hd

def cal_metrics(y_true, y_pred):

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1).astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1).astype(np.uint8)

    ## Score
    jac_miou = miou_score(y_true, y_pred)
    f1_dice = dice_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f2 = F2_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    # specificity = 0
    
    return [jac_miou, f1_dice, recall, precision, specificity, acc, f2]



