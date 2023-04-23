import torch
import torch.nn as nn
import torch.nn.functional as F

class KD_loss(nn.Module):
    def __init__(self, Temperature=20):
        """
            input:
                Temperature - "To make the softmax outputs smoother thus distill \
                    the knowledge of label distribution from teacher's prediction"
        """
        super(KD_loss,self).__init__()
        self.T = Temperature
    
    def forward(self, z, r, y):
        """
            input : 
                y : (gt)
                y_stu : (student output)
                y_tea : (teacher output)
                
            output : 
                loss (Variable) : 논문's distillation loss
        """
        default_loss = nn.CrossEntropyLoss()(z,y)        # TODO How this could work? --> "default_loss"  be an insatnce carrying some needed values.
        term1 = F.softmax(torch.mul(r,1/self.T))         # nn.functional 이 softmax의 computational graph를 지원하나?
        term2 = F.softmax(torch.mul(z,1/self.T))
        distill_loss = self.T**2 * nn.CrossEntropyLoss()(term1, term2)

        loss = default_loss + distill_loss

        return loss