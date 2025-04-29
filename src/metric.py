from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('tp', default=torch.zeros(200), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(200), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(200), dist_reduce_fx='sum')

    
    def update(self, preds, target):
        # check if preds and target have equal shape
        if preds.shape[0] != target.shape[0]:
            raise ValueError(f"Preds shape {preds.shape} and target shape {target.shape} must be equal")

        max_idx = preds.argmax(dim=-1)

        for i in range(preds.shape[0]):
            pred_class = max_idx[i]
            gt_class = target[i]

            if pred_class == gt_class: # increment tp 1
                self.tp[pred_class] += 1
            else: # increment fp and fn 1
                self.fp[pred_class] += 1
                self.fn[gt_class] += 1


    def compute(self):
        # Calculate precision and recall
        precision = self.tp.float() / (self.tp + self.fp).float()
        recall = self.tp.float() / (self.tp + self.fn).float()

        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall)

        # filter out except nan values
        f1_score = f1_score[~torch.isnan(f1_score)]
        if f1_score.shape[0] == 0:
            return torch.zeros(1)
        else:
            return f1_score.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        max_idx = preds.argmax(dim=-1)

        # [TODO] check if preds and target have equal shape
        if max_idx.shape != target.shape:
            raise ValueError(f"Preds shape {max_idx.shape} and target shape {target.shape} must be equal")


        # [TODO] Count the number of correct prediction
        correct = (max_idx == target).sum()


        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

