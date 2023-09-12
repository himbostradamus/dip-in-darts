import numpy as np
import torch.nn as nn

class ES(nn.Module):
    def __init__(self, 
                 buffer_size=100,
                 patience=1000,
                ):
        super().__init__()
        
        # early stop criteria
        self.img_collection = []
        self.variance_history = []
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.burnin_over = False
        self.buffer_size = buffer_size
        self.cur_var = None

    def update_stop(self,out_np):
        """
        Componenet of closure function
        check if we should end the burnin phase
        """
        # update img collection
        v_img_np = out_np.reshape(-1)
        self.update_img_collection(v_img_np)
        img_collection = self.get_img_collection()

        if len(img_collection) >= self.buffer_size:
            # update variance and var history
            ave_img = np.mean(img_collection, axis=0)
            variance = [self.MSE(ave_img, tmp) for tmp in img_collection]
            self.cur_var = np.mean(variance)
            self.variance_history.append(self.cur_var)
            self.check_stop(self.cur_var, self.i)

    def check_stop(self, current, cur_epoch):
        """
        using an early stopper technique to determine when to end the burn in phase for SGLD
        https://arxiv.org/pdf/2112.06074.pdf
        https://github.com/sun-umn/Early_Stopping_for_DIP/blob/main/ES_WMV.ipynb
        """
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            self.burnin_over = False
        else:
            self.wait_count += 1
            self.burnin_over = self.wait_count >= self.patience
        if self.burnin_over:
            print(f'\n\nBurn-in completed at iter {self.i}; \nStarting SGLD Mean sampling;\n\n')
            # self.show_every = self.MCMC_iter

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.buffer_size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

    def MSE(self, x1, x2):
        return ((x1 - x2) ** 2).sum() / x1.size()[0]