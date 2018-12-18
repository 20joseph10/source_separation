import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.25, num_layers=3):
        super(BaselineModel, self).__init__()
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        self.rnn_cell1 = nn.GRUCell(input_size, hidden_size)
        self.rnn_cell2 = nn.GRUCell(hidden_size, hidden_size)
        self.rnn_cell3 = nn.GRUCell(hidden_size, hidden_size)
        self.num_layers = num_layers
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        # self.bn_rnn2 = nn.BatchNorm1d(hidden_size)
        # self.bn_rnn3 = nn.BatchNorm1d(hidden_size)
        # self.c = torch.randn(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, input_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.init_weights()
        
    def init_weights(self):
        """
        randomly initilize weights for linear layers
        """
        init_value = 0.1
        self.linear1.weight.data.uniform_(-init_value, init_value)
        self.linear2.weight.data.uniform_(-init_value, init_value)

    def forward(self, x):
        

        # output, states = self.rnn(x)
        output = []
        
        for i in range(x.shape[1]):

            if i == 0:
                h1 = self.rnn_cell1(x[:,i,:])
                h1 = self.drop1(h1)
                h2 = self.rnn_cell2(h1)
                h2 = self.drop2(h2)
                h3 = self.rnn_cell3(h2)
            else:
                h1 = self.rnn_cell1(x[:,i,:], h1)
                h1 = self.drop1(h1)
                h2 = self.rnn_cell2(h1, h2)
                h2 = self.drop2(h2)
                h3 = self.rnn_cell3(h2, h3)

            output.append(h3)

        output = torch.stack(output).permute(1,0,2)

        s1 = F.relu(self.linear1(output))
        s2 = F.relu(self.linear2(output))

        # soft time frequency mask
        mask1 = torch.abs(s1) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        mask2 = torch.abs(s2) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        # # mask1 = self.linear_mask1(mask1) + mask1
        # # mask2 = self.linear_mask2(mask2) + mask2 
        s1 = mask1*x
        s2 = mask2*x
        return s1, s2
class SuperModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.4, num_layers=3):
        super(SuperModel, self).__init__()
        self.rnn = nn.GRU(382+512, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        # self.rnn_cell1 = nn.GRUCell(input_size, hidden_size)
        # self.rnn_cell2 = nn.GRUCell(hidden_size, hidden_size)
        # self.rnn_cell3 = nn.GRUCell(hidden_size, hidden_size)
        self.num_layers = num_layers
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    

        self.cnn_block1 = self.conv_block(1, 64, (1,3), dropout)
        self.cnn_block2 = self.conv_block(64, 64, (1,3), dropout)
        # self.cnn_block3 = self.conv_block(64, 64, (1,3), dropout)
        
        self.rnn_conv1 = nn.Conv2d(64, 1, 1)
        self.rnn_conv2 = nn.Conv2d(64, 1, 1)
        self.linear1 = nn.Linear(894, input_size)
        self.linear2 = nn.Linear(894, input_size)
        self.init_weights()

    def conv_block(self, in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1,2)),
            nn.PReLU(),
            nn.Dropout(dropout))

    def init_weights(self):
        """
        randomly initilize weights for linear layers
        """
        init_value = 0.1
        self.linear1.weight.data.uniform_(-init_value, init_value)
        self.linear2.weight.data.uniform_(-init_value, init_value)

    def forward(self, x):
        # feature extraction
        conv_input = x.unsqueeze(1)
        # conv_input = self.pool1(conv_input)
        # print(conv_input.shape)
        feat1 = self.cnn_block1(conv_input)
        feat2 = self.cnn_block2(feat1)
        # feat3 = self.cnn_block3(feat2)
        # feat4 = self.cnn_block4(feat3)
        # feat5 = self.cnn_block5(feat4)
        # print(feat3.shape)
        # concat
        conv_cat = torch.cat((feat1, feat2, ), 3)
        conv_cat_ = self.rnn_conv1(conv_cat)
        
        # print(conv_cat.shape)
        rnn_in = torch.cat((conv_cat_.squeeze(1), x), 2)

        # print(rnn_in.shape)
        # rnn_in = self.rnn_conv1(feat5)
        # print(rnn_in.shape)
        output, states = self.rnn(rnn_in)
        conv_out = self.rnn_conv2(conv_cat).squeeze(1)
        final_cat = torch.cat((output, conv_out), 2)
        # print(final_cat.shape)
        # print(conv_out.shape)
        # print(output.shape)
        s1 = F.relu(self.linear1(final_cat))
        s2 = F.relu(self.linear2(final_cat))

        # soft time frequency mask
        mask1 = torch.abs(s1) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        mask2 = torch.abs(s2) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        # # mask1 = self.linear_mask1(mask1) + mask1
        # # mask2 = self.linear_mask2(mask2) + mask2 
        s1 = mask1*x
        s2 = mask2*x
        return s1, s2
class CNNRNNCNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.25):
        super(CNNRNNCNNModel, self).__init__()
        self.rnn = nn.GRU(252, 512, num_layers=3, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, input_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        
        # conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv2_1x1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.conv2_1x1_2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1)
        self.rnn_out_linear = nn.Linear(512, 252)
        self.conv2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=3)
        
        

    def forward(self, x):
        # (batch size, channel 1, freq, time)
        conv_input = x.permute(0,2,1).unsqueeze(1) #torch.Size([64, 1, 513, 64])

        conv1 =  self.pool1(conv_input) # torch.Size([64, 1, 256, 32])
        conv1 = F.relu(self.conv1(conv1)) #         torch.Size([64, 64, 254, 30])
        # conv1 = F.dropout(conv1, training=self.training)
        conv2 = F.relu(self.conv2_1(conv1))#         torch.Size([64, 128, 252, 28])
        conv1_1 = F.relu(self.conv2_1x1(conv2))
        # conv1_1 = F.dropout(conv1_1, training=self.training)
        rnn_in = conv1_1.squeeze(1).permute(0,2,1)
        output, _ = self.rnn(rnn_in)
        output = self.rnn_out_linear(output) # 128 to 252
        output = output.permute(0,2,1).unsqueeze(1)
        output = F.dropout(output, p=0.2, training=self.training)

        output = F.relu(self.conv2_1x1_2(output)) # add relu output: 64, 64, 252 28
        # output = F.dropout(output, training=self.training)
        conv2 = conv2 + output
        conv2 = F.relu(self.conv2_2(conv2)) # add relu
        conv2 = F.dropout(conv2, p=0.2, training=self.training)
        
        conv3 = F.relu(self.conv3(conv2)) # add relu
        # conv3 = F.dropout(conv3, training=self.training)
        conv3 = F.interpolate(conv3, scale_factor=2)
        
        output = conv3.squeeze(1).permute(0,2,1)
        #print(output.shape)
        s1 = F.relu(self.linear1(output))
        s2 = F.relu(self.linear2(output))

        # soft time frequency mask
        mask1 = torch.abs(s1) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        mask2 = torch.abs(s2) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        # # mask1 = self.linear_mask1(mask1) + mask1
        # # mask2 = self.linear_mask2(mask2) + mask2 
        s1 = mask1*x
        s2 = mask2*x
        return s1, s2

class CNNRNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.25):
        super(CNNRNNBaseline, self).__init__()
        self.rnn = nn.GRU(512, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, input_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        # self.linear_mask1 = nn.Linear(input_size, input_size)
        # self.linear_mask2 = nn.Linear(input_size, input_size)
        
        # conv
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3))
        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(3,3))

        
        

    def forward(self, x):
        # print(x.shape)
        # (batch size, channel 1, freq, time)
        conv_input = x.permute(0,2,1).unsqueeze(1) #torch.Size([64, 1, 512, 64])
        conv1 = self.pool1(conv_input) # 64, 1, 256, 32
        # print(conv1.shape)
        conv1 = F.relu(self.conv1(conv1)) # 64, 256, 254, 30
        conv2 = F.relu(self.conv2(conv1)) # 64, 1, 256, 32
        conv2 = F.interpolate(conv2, scale_factor=(2,2)) # 64, 1, 512, 64
        rnn_in = conv2.squeeze(1).permute(0,2,1) # 64, 64, 512
        # print(rnn_in.shape)
        # idea, fn to combine original input and learned feature?
        output, _ = self.rnn(rnn_in) # 64, 64, hidden_size
        
        # output = torch.stack(output).permute(1,0,2) # two ways to add skip, additive or concatenative
        s1 = F.relu(self.linear1(output)) # 64, 64, 512
        s2 = F.relu(self.linear2(output)) # 64, 64, 512

        # soft time frequency mask
        mask1 = torch.abs(s1) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        mask2 = torch.abs(s2) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        
        s1 = mask1*x
        s2 = mask2*x
        return s1, s2

class EncoDecoderModelv2(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.25):
        super(EncoDecoderModelv2, self).__init__()
        self.rnn = nn.GRU(512, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, input_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        # self.dropout = nn.Dropout(dropout)

        self.rnn_cell1 = nn.GRUCell(input_size, hidden_size)
        self.rnn_cell2 = nn.GRUCell(hidden_size, hidden_size)
        self.rnn_cell3 = nn.GRUCell(hidden_size, hidden_size)

        # self.h = torch.randn(input_size, hidden_size)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        # self.linear_mask1 = nn.Linear(input_size, input_size)
        # self.linear_mask2 = nn.Linear(input_size, input_size)
        
        # conv
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3))
        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(3,3))
        self.linear_cnn_x = nn.Linear(1024, 512)
        self.linear_cnn_drop = nn.Dropout(dropout)
        

    def forward(self, x):
        # print(x.shape)
        # (batch size, channel 1, freq, time)
        # print(x.shape)
        conv_input = x.permute(0,2,1).unsqueeze(1) #torch.Size([64, 1, 512, 64])
        conv1 = self.pool1(conv_input) # 64, 1, 256, 32
        # print(conv1.shape)
        conv1 = F.relu(self.conv1(conv1)) # 64, 256, 254, 30
        conv2 = F.relu(self.conv2(conv1)) # 64, 1, 256, 32
        conv2 = F.interpolate(conv2, scale_factor=(2,2)) # 64, 1, 512, 64
        conv_out = conv2.squeeze(1).permute(0,2,1)
        cnn_out = self.linear_cnn_drop(conv_out)
        cnn_out = torch.cat((cnn_out, x), 2)
        rnn_in = F.relu(self.linear_cnn_x(cnn_out))
        
        output, _ =self.rnn(rnn_in)
        
        s1 = F.relu(self.linear1(output)) # 64, 64, 512
        s2 = F.relu(self.linear2(output)) # 64, 64, 512


        # soft time frequency mask
        mask1 = torch.abs(s1) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        mask2 = torch.abs(s2) / (torch.abs(s1)+torch.abs(s2)+1e-16)
        
        s1 = mask1*x
        s2 = mask2*x
        return s1, s2
    
    

def time_freq_masking(M_stft, L_hat, S_hat, gain=3):
    # mask = np.abs(S_hat) - gain * np.abs(L_hat)
    # mask = (mask > 0) * 1
    # # print(mask)
    # X_sing = np.multiply(mask, M_stft)
    # X_music = np.multiply(1 - mask, M_stft)
    # return X_sing, X_music
    pass
class R_pca:
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')


def pcp_alm(X, maxiter=500, tol=1e-7, gamma_spec=True):
    """
    rpca algorithm

    Principal Component Pursuit
    Finds the Principal Component Pursuit solution.
    Solves the optimization problem::
        (L^*,S^*) = argmin || L ||_* + gamma * || S ||_1
                    (L,S)
                    subject to    L + S = X
    where || . ||_* is the nuclear norm.  Uses an augmented Lagrangian approach
    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        Data matrix.
    maxiter : int, 500 by default
        Maximum number of iterations to perform.
    tol : float, 1e-7 by default - in the paper

    gamma_spec : True or a float. If gamma_spec = True, then algorithm gamma = 1/sqr(max_dimension), else specify a gamma parameter in float.

    Returns
    -------
    L : array of shape [n_samples, n_features]
        The low rank component
    S : array of shape [n_samples, n_features]
        The sparse component
    (u, sig, vt) : tuple of arrays
        SVD of L.
    n_iter : int
        Number of iterations
    Reference
    ---------
       Candes, Li, Ma, and Wright
       Robust Principal Component Analysis?
       Submitted for publication, December 2009.
    """

    def soft_threshold(v=0, tau=0):
        '''
        shrinkiage opterator S_tau[x]
        '''
        tmp = np.abs(v)
        tmp = np.subtract(tmp, tau)
        tmp = np.maximum(tmp, 0.0)
        return np.multiply(np.sign(v), tmp)

    def svt(X, tau, k, svd_function, out=None):
        def svd_reconstruct(u, sig, v, out=None, tmp_out=None):
            tmp = np.multiply(u, sig, out=tmp_out)
            return np.dot(tmp, v, out=out)

        def truncate_top_k_svd(u, sig, v, k):
            return u[:, :k], sig[:k], v[:k, :]

        m, n = X.shape

        u, sig, v = svd_function(X, k)

        sig = soft_threshold(sig, tau)
        r = np.sum(sig > 0)

        u, sig, v = svd_function(X, r)
        sig = soft_threshold(sig, tau)

        if r > 0:
            #print("Z= reconstructed")
            u, sig, v = truncate_top_k_svd(u, sig, v, r)
            #print("reconstructed sig =", sig)
            Z = svd_reconstruct(u, sig, v, out=out)
        else:
            #print("Z= 0")
            out[:] = 0
            Z = out
            u, sig, v = np.empty((m, 0)), np.empty((0, 0)), np.empty((0, n))
        return (Z, r, (u, sig, v))

    def svd_choice(n, d):
        '''
        choose svd depend on the size 

        return 'dense_top_k_svd/sparse_svds
        '''

        ratio = float(d) / float(n)
        vals = [(0, 0.02), (100, 0.06), (200, 0.26),
                (300, 0.28), (400, 0.34), (500, 0.38)]

        i = bisect.bisect_left([r[0] for r in vals], n)
        choice = dense_top_k_svd if ratio > vals[i - 1][1] else svds
        return choice

    def dense_top_k_svd(A, k):
        '''
        A - matrix
        k - Top K components
        '''
        u, sig, v = svd(A, full_matrices=0)
        return u[:, :k], sig[:k], v[:k, :]

    n = X.shape
    frob_norm = np.linalg.norm(X, 'fro')
    two_norm = np.linalg.norm(X, 2)
    one_norm = np.sum(np.abs(X))
    inf_norm = np.max(np.abs(X))

    mu_inv = 4 * one_norm / np.prod(n)

    if gamma_spec:
        gamma = 1 / np.sqrt(np.max([n[0], n[1]]))
    else:
        gamma = gamma_spec
    k = np.min([
        np.floor(mu_inv / two_norm),
        np.floor(gamma * mu_inv / inf_norm)
    ])
    Y = k * X
    sv = 10

    # Variable init
    S = np.zeros(n)

    # print("k",k)
    #print("mu_inv", mu_inv)

    for i in range(maxiter):

        # Shrink singular values
        l = X - S + mu_inv * Y
        svd_fun = svd_choice(np.min(l.shape), sv)
        # print(svd_fun)
        #print("sv", sv)
        L, r, (u, sig, v) = svt(l, mu_inv, sv, svd_function=svd_fun)
        #print("non-zero sigular value", r)
        if r < sv:
            sv = np.min([r + 1, np.min(n)])
        else:
            sv = np.min([r + int(np.round(0.05 * np.min(n))), np.min(n)])

        # Shrink entries
        s = X - L + mu_inv * Y
        S = soft_threshold(s, gamma * mu_inv)

        # Check convergence
        R = X - L - S
        stopCriterion = np.linalg.norm(R, 'fro') / frob_norm
        #print("stopCriterion", stopCriterion)
        if stopCriterion < tol:
            break

        # Update dual variable
        Y += 1 / mu_inv * (X - L - S)

    return L, S, (u, sig, v), i + 1


def separate_signal_with_RPCA(M_mag, M_phase, improve=False, gamma_spec=True):
    # Short-Time Fourier Transformation
    # M_stft = librosa.stft(M, n_fft=1024, hop_length=256)
    # Get magnitude and phase
    # RPCA
    L_hat, S_hat, r_hat, n_iter_hat = pcp_alm(M_mag, gamma_spec=gamma_spec)
    # Append phase back to result
    L_output = np.multiply(L_hat, M_phase)
    S_output = np.multiply(S_hat, M_phase)

    if improve:
        L_hat, S_hat, r_hat, n_iter_hat = pcp_alm(np.abs(S_output))
        S_output = np.multiply(S_hat, M_phase)
        L_output = M_stft - S_output

    return L_output, S_output
