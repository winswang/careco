import numpy as np

class lp_precomp():
    def __init__(self, trs_res):
        self.row, self.col = trs_res
        
        dn = 1/float(np.min((self.col, self.row)))
        log_rho_axis = np.linspace(np.log(dn), 0, self.col)
        theta_axis = np.linspace(0, np.pi*(1-dn), self.row)
        x0 = self.col/2.0 + 1
        y0 = self.row/2.0 + 1
        rmax = np.sqrt(x0**2 + y0**2)
        lp_table_x = np.empty((self.row, self.col))
        lp_table_y = np.empty((self.row, self.col))
        self.mask = np.ones((self.row, self.col))
        for ii in range(self.row):
            for jj in range(self.col):
                rho = np.exp(log_rho_axis[jj])
                theta = theta_axis[-ii-1]
                xx = x0 + rho * np.cos(theta) * rmax
                yy = self.row - y0 - rho * np.sin(theta) * rmax
                if xx < 0 or xx > (self.col - 1) or yy < 0 or yy > (self.row - 1):
                    self.mask[ii,jj] = 0
                    lp_table_x[ii,jj] = 0
                    lp_table_y[ii,jj] = 0
                else:
                    lp_table_x[ii,jj] = xx
                    lp_table_y[ii,jj] = yy
        lp_table_x = lp_table_x.flatten()
        lp_table_y = lp_table_y.flatten()
        cleft = np.floor(lp_table_x).astype(int)
        cright = cleft + 1
        ax = lp_table_x - cleft
        cup = np.floor(lp_table_y).astype(int)
        cdown = cup + 1
        ay = lp_table_y - cup
        self.iul = cup*self.col + cleft
        self.iur = cup*self.col + cright
        self.idl = cdown*self.col + cleft
        self.idr = cdown*self.col + cright
        self.aul = np.multiply((1-ax), (1-ay))
        self.aur = np.multiply(ax, (1-ay))
        self.adl = np.multiply((1-ax), ay)
        self.adr = np.multiply(ax, ay)
        self.mask = self.mask.flatten()
        
class trs():
    def __init__(self, op_res = None, lp_precomp = None):
        if op_res == None:
            self.row = 256
            self.col = 256
        else:
            self.row, self.col = op_res
        if lp_precomp == None:
            self.lp_pc_flag = 0
        else:
            self.lp_pc_flag = 1
            self.iul = lp_precomp.iul
            self.iur = lp_precomp.iur
            self.idl = lp_precomp.idl
            self.idr = lp_precomp.idr
            self.aul = lp_precomp.aul
            self.aur = lp_precomp.aur
            self.adl = lp_precomp.adl
            self.adr = lp_precomp.adr
            self.mask = lp_precomp.mask
        
    def norm_max(self, img):
        return img/np.amax(img)
    
    def phase_corr(self,img1,img2):
        self.img1_fft_t = np.fft.fftshift(np.fft.fft2(img1))
        self.img2_fft_t = np.fft.fftshift(np.fft.fft2(img2))
        temp_dot = np.multiply(self.img1_fft_t, np.conj(self.img2_fft_t))
        cps = temp_dot/(np.absolute(temp_dot)+np.finfo(float).eps)
        cps_ifft = np.nan_to_num(np.absolute(np.fft.fftshift(np.fft.ifft2(cps))))
        return cps_ifft
    
    def log_polar(self, img):
        img_flat = img.flatten()
        if self.lp_pc_flag == 0:
            dn = 1/float(np.min((self.col, self.row)))
            log_rho_axis = np.linspace(np.log(dn), 0, self.col)
            theta_axis = np.linspace(0, np.pi*2*(1-dn), self.row)
            x0 = self.col/2.0 + 1
            y0 = self.row/2.0 + 1
            rmax = np.sqrt(x0**2 + y0**2)
            lp_table_x = np.empty((self.row, self.col))
            lp_table_y = np.empty((self.row, self.col))
            self.mask = np.ones((self.row, self.col))
            for ii in range(self.row):
                for jj in range(self.col):
                    rho = np.exp(log_rho_axis[jj])
                    theta = theta_axis[-ii-1]
                    xx = x0 + rho * np.cos(theta) * rmax
                    yy = self.row - y0 - rho * np.sin(theta) * rmax
                    if xx < 0 or xx > (self.col - 1) or yy < 0 or yy > (self.row - 1):
                        self.mask[ii,jj] = 0
                        lp_table_x[ii,jj] = 0
                        lp_table_y[ii,jj] = 0
                    else:
                        lp_table_x[ii,jj] = xx
                        lp_table_y[ii,jj] = yy
            lp_table_x = lp_table_x.flatten()
            lp_table_y = lp_table_y.flatten()
            cleft = np.floor(lp_table_x).astype(int)
            cright = cleft + 1
            ax = lp_table_x - cleft
            cup = np.floor(lp_table_y).astype(int)
            cdown = cup + 1
            ay = lp_table_y - cup
            self.iul = cup*self.col + cleft
            self.iur = cup*self.col + cright
            self.idl = cdown*self.col + cleft
            self.idr = cdown*self.col + cright
            self.aul = np.multiply((1-ax), (1-ay))
            self.aur = np.multiply(ax, (1-ay))
            self.adl = np.multiply((1-ax), ay)
            self.adr = np.multiply(ax, ay)
            self.mask = self.mask.flatten()
            
            self.lp_pc_flag = 1

        canvas = np.multiply(self.mask, np.multiply(img_flat[self.iul], self.aul) + np.multiply(img_flat[self.iur], self.aur) + np.multiply(img_flat[self.idl], self.adl) + np.multiply(img_flat[self.idr], self.adr))    
 
        return canvas.reshape(self.row, self.col)

    def trs_pair(self,img1,img2):
        tmap = self.phase_corr(img1,img2)
        self.img1_lp = self.log_polar(np.absolute(self.img1_fft_t))
        self.img2_lp = self.log_polar(np.absolute(self.img2_fft_t))
        rsmap = self.phase_corr(self.img1_lp,self.img2_lp)
        return tmap, rsmap