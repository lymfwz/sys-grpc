import torch
import numpy as np


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    # corr=0
    SR = SR > threshold
    # SR = (float) (SR >threshold)
    # zero = torch.zeros_like(GT)
    one = torch.ones_like(GT)
    GT = torch.where(GT > 0, one, GT)
    # GT = torch.where(GT <= 0, zero, GT)
    corr = torch.sum(SR == GT)
    # print(SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3))
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)
    # TP=0
    # TN=0
    # FN=0
    # FP=0
    # SR = SR > threshold
    # one = torch.ones_like(GT)
    # zero = torch.zeros_like(GT)
    # SR = torch.where(SR > 0.5, one, SR)
    # SR = torch.where(SR < 0.5, zero, SR)
    # GT = GT == torch.max(GT)
    # GT = torch.where(GT > 0, one, GT)

    # TP += ((SR == 1) & (GT == 1)).cpu().sum()

    # TN += ((SR == 0) & (GT== 0)).cpu().sum()
    # FN    predict 0 label 1
    # FN += ((SR == 0) & (GT == 1)).cpu().sum()
    # FP    predict 1 label 0
    # FP += ((SR == 1) & (GT == 0)).cpu().sum()

    # acc = (TP + TN) / (TP + TN + FP + FN)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    TP = 0
    FN = 0
    # Sensitivity == Recall
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    # one = torch.ones_like(GT)
    # zero = torch.zeros_like(GT)
    one = torch.ones_like(GT)
    # zero = torch.zeros_like(GT)
    # SR = torch.where(SR > 0.5, one, SR)
    # SR = torch.where(SR < 0.5, zero, SR)
    GT = torch.where(GT > 0, one, GT)

    # GT = torch.where(GT < 0.5, zero, GT)

    # TP : True Positive
    # FN : False Negative
    # train_correct01 = float(torch.sum(((SR == zero) & (GT == one))))  # 原标签为1，预测为 0 的总数
    # train_correct10 = torch.sum(((SR == one) & (GT == zero)))   # 原标签为0，预测为1 的总数
    # train_correct11 = torch.sum((SR == one) & (GT == one)).sum()
    # FN += train_correct01.data[0]
    # FP += train_correct10.data[0]
    # TP += train_correct11.data[0]
    # TN += train_correct00.data[0]
    # TP = ((SR==1.)+(GT==1.))==2
    TP += ((SR == 1) & (GT == 1)).cpu().sum()
    # FN = ((SR==0.)+(GT==1.))==2
    FN += ((SR == 0) & (GT == 1)).cpu().sum()
    # SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    # SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) )
    SE = TP / (TP + FN + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    # one = torch.ones_like(GT)
    one = torch.ones_like(GT)
    # zero = torch.zeros_like(GT)
    # SR = torch.where(SR > 0.5, one, SR)
    # SR = torch.where(SR < 0.5, zero, SR)
    GT = torch.where(GT > 0, one, GT)

    # TN : True Negative
    # FP : False Positive
    # TN = ((SR==0)+(GT==0))==2
    # FP = ((SR==1)+(GT==0))==2
    # TP=0
    TN = 0
    # FN=0
    FP = 0
    # Sensitivity == Recall
    # SR = SR > threshold
    # GT = GT == torch.max(GT)
    # one = torch.ones_like(GT)
    # zero = torch.zeros_like(GT)
    # GT = torch.where(GT > 0, one, GT)

    # TP += ((SR == 1) & (GT == 1)).cpu().sum()
    # TN    predict �� label ͬʱΪ0
    TN += ((SR == 0) & (GT == 0)).cpu().sum()
    # FN    predict 0 label 1
    # FN += ((SR == 0) & (GT == 1)).cpu().sum()
    # FP    predict 1 label 0
    FP += ((SR == 1) & (GT == 0)).cpu().sum()

    # SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    SP = TN / (TN + FP + 1e-6)
    # SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)))

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    one = torch.ones_like(GT)
    # zero = torch.zeros_like(GT)
    # SR = torch.where(SR > 0.5, one, SR)
    # SR = torch.where(SR < 0.5, zero, SR)
    # one = torch.ones_like(GT)
    GT = torch.where(GT > 0, one, GT)

    # TP : True Positive
    # FP : False Positive
    # TP = ((SR==1)+(GT==1))==2
    # FP = ((SR==1)+(GT==0))==2
    TP = 0
    # TN=0
    # FN=0
    FP = 0

    TP += ((SR == 1) & (GT == 1)).cpu().sum()
    # TN    predict �� label ͬʱΪ0
    # TN += ((SR == 0) & (GT== 0)).cpu().sum()
    # FN    predict 0 label 1
    # FN += ((SR == 0) & (GT == 1)).cpu().sum()
    # FP    predict 1 label 0
    FP += ((SR == 1) & (GT == 0)).cpu().sum()

    PC = TP / (TP + FP + 1e-6)
    # PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) )

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)
    # F1 = 2 * SE * PC / (SE + PC )

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    one = torch.ones_like(GT)
    # GT = GT == torch.max(GT)
    GT = torch.where(GT > 0, one, GT)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)
    # JS = float(Inter) / (float(Union))

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    one = torch.ones_like(GT)
    # GT = GT == torch.max(GT)
    GT = torch.where(GT > 0, one, GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)
    # DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)))

    return DC