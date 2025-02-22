import torch
import matplotlib.pyplot as plt


def calculate_tpr_and_fpr(p, true_labels, threshold):
    # Classification based on threshold
    q = torch.where(p <= threshold, torch.floor(p).bool(), torch.ceil(p).bool())
    tp = q.multiply(true_labels).count_nonzero()
    tn = q.bitwise_not().multiply(true_labels.bitwise_not()).count_nonzero()
    fp = q.multiply(true_labels.bitwise_not()).count_nonzero()
    fn = q.bitwise_not().multiply(true_labels).count_nonzero()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr
    
    


def train():
    pred = torch.tensor([0.7, 0.6, 0.5, 0.2, 0.3, 0.9, 0.7], dtype=torch.float32)
    true_labels = torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    xlim = []
    ylim = []
    for i in range(11):
        threshold = 0.1*i
        tpr, fpr = calculate_tpr_and_fpr(pred, true_labels, threshold)
        xlim.append(fpr)
        ylim.append(tpr)
    print(xlim)
    print(ylim)
    plt.figure()  
    plt.plot(xlim, ylim, label= 'AUC')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

if __name__ == "__main__":
    train()
    