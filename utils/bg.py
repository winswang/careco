import numpy as np

class background():
    def __init__(self, op_res = None):
        if op_res == None:
            self.row = 256
            self.col = 256
            self.frames = 10
        else:
            (self.row, self.col, self.frames) = op_res
            
    def med_filter(self, vid, kvar):
        med = np.median(vid, axis = 2)
        alpha = np.empty((self.row, self.col, self.frames))
        for i in range(self.frames):
            alpha[:,:,i] = 1 - np.exp(-np.square(vid[:,:,i] - med)/kvar)
        return np.multiply(vid, alpha)

        
        
