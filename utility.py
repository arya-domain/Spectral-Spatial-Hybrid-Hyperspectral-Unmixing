import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable


class HSI:
    def __init__(self, data, rows, cols, gt, abundance_gt):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        self.image = np.reshape(data, (self.rows, self.cols, self.bands))
        self.gt = gt
        self.abundance_gt = abundance_gt

    def array(self):
        return np.reshape(self.image, (self.rows * self.cols, self.bands))


def load_HSI(path):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')

    numpy_array = np.asarray(data['Y'], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()

    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None

    if 'S_GT' in data.keys():
        abundance_gt = np.asarray(data['S_GT'], dtype=np.float32)
    else:
        abundance_gt = None

    return HSI(numpy_array, n_rows, n_cols, gt, abundance_gt)


def numpy_MSE(y_true, y_pred):
    num_cols = y_pred.shape[0]
    num_rows = y_pred.shape[1]
    diff = y_true - y_pred
    squared_diff = np.square(diff)
    mse = squared_diff.sum() / (num_rows * num_cols)
    rmse = np.sqrt(mse)
    return rmse


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance


def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos > 1.0: cos = 1.0
    return np.arccos(cos)


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    SAD_matrix = np.zeros((num_endmembers, num_endmembers))
    SAD_index = np.zeros(num_endmembers).astype(int)
    SAD_endmember = np.zeros(num_endmembers)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    e = endmembers.copy()
    egt = endmembersGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            SAD_matrix[i, j] = numpy_SAD(e[i, :], egt[j, :])

        SAD_index[i] = np.nanargmin(SAD_matrix[i, :])
        SAD_endmember[i] = np.nanmin(SAD_matrix[i, :])
        egt[SAD_index[i], :] = np.inf
    return SAD_index, SAD_endmember

import seaborn as sns
def plotEndmembersAndGT(endmembers, endmembersGT, endmember_path, sadsave):
    sns.set_style("white")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2) + (num_endmembers % 2)
    
    SAD_index, SAD_endmember = order_endmembers(endmembersGT, endmembers)
    mean_sad = SAD_endmember.mean()
    title = f"mSAD: {mean_sad:.3f} radians"
    print(title)
    endmembers = endmembers / endmembers.max(axis=1, keepdims=True)
    endmembersGT = endmembersGT / endmembersGT.max(axis=1, keepdims=True)

    fig, axs = plt.subplots(2, n, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, weight='bold', color="#2f4f4f", y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.4, hspace=0.6)
    
    linewidth = 3
    fontsize = 25

    for i in range(num_endmembers):
        ax = axs.flat[i] if num_endmembers > 1 else axs
        ax.plot(endmembersGT[i, :], color='#f80000', linewidth=linewidth, label='Ground Truth') # '#2e8b57'
        ax.plot(endmembers[SAD_index[i], :], color='#000000', linewidth=linewidth, label='Predicted') # '#ff6347'
        sad_value = numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :])
        # ax.set_title(f"SAD: {sad_value:.3f} for Class {i+1}", fontsize=fontsize, pad=10, fontweight='bold')
        ax.get_xaxis().set_visible(False)
        # ax.legend(loc="best", fontsize=30, frameon=True, prop={'weight': 'bold'})
        sadsave.append(sad_value)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')  # Set the color of the border
            spine.set_linewidth(3) 

    sadsave.append(mean_sad)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{endmember_path}.png", bbox_inches="tight")


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance


def plotAbundancesSimple(abundances, abundanceGT, abundance_path, rmsesave):
    sns.set_style("white")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2 + (num_endmembers % 2)
    
    abundance_index, MSE_abundance = order_abundance(abundanceGT, abundances)
    mean_rmse = MSE_abundance.mean()
    title = f"RMSE: {mean_rmse:.3f}"
    print(title)
    # cmap = 'jet' viridis
    cmap = 'jet'
    
    fig, axs = plt.subplots(2, n, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, weight='bold', color="#2f4f4f", y=0.98)
    fig.subplots_adjust(top=0.9, wspace=0.4, hspace=0.4)
    
    for i in range(num_endmembers):
        ax = axs.flat[i] if num_endmembers > 1 else axs
        im = ax.imshow(abundances[:, :, abundance_index[i]], cmap=cmap)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, orientation='vertical')
        
        rmse_value = numpy_MSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i])
        ax.set_title(f"RMSE: {rmse_value:.3f} for Class {i+1}", fontsize=16, pad=10)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        rmsesave.append(rmse_value)

    rmsesave.append(mean_rmse)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{abundance_path}.png", bbox_inches="tight")



def plotAbundancesGT(abundanceGT, abundance_path):
    num_endmembers = abundanceGT.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    title = 'Abundance GT'
    cmap = 'viridis'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundanceGT, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundanceGT[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.rcParams.update({'font.size': 19})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()


def reconstruct(S, A):
    S = np.reshape(S, (S.shape[0] * S.shape[1], S.shape[2]))
    reconstructed = np.matmul(S, A)
    return reconstructed


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, abundance_gt, transform=None):
        self.img = img.float()
        self.transform = transform
        self.abundance_gt = abundance_gt
        
    def __getitem__(self, idx):
        return self.img, self.abundance_gt

    def __len__(self):
        return 1


def pca(X, d):
    N = np.shape(X)[1]
    xMean = np.mean(X, axis=1, keepdims=True)
    XZeroMean = X - xMean
    [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
    Ud = U[:, 0:d]
    return Ud


def hyperVca(M, q):
    '''
    M : [L,N]
    '''
    L, N = np.shape(M)

    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, 0:q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    SNRth = 18 + 10 * np.log(q)

    if SNR > SNRth:
        d = q
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        Y = Xd / np.sum(Xd * u, axis=0, keepdims=True)

    else:
        d = q - 1
        r_bar = np.mean(M.T, axis=0, keepdims=True).T
        Ud = pca(M, d)

        R_zeroMean = M - r_bar
        Xd = Ud.T @ R_zeroMean
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([q, 1])
    e_u[q - 1, 0] = 1
    A = np.zeros([q, q])
    A[:, 0] = e_u[0]
    I = np.eye(q)
    k = np.zeros([N, 1])

    indicies = np.zeros([q, 1])
    for i in range(q):  # i=1:q
        w = np.random.random([q, 1])
        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        f = tmpNumerator / np.linalg.norm(tmpNumerator)

        v = f.T @ Y
        k = np.abs(v)

        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    if (SNR > SNRth) :
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U, indicies, snrEstimate


def SAD(y_true, y_pred):
    y_true2 = torch.nn.functional.normalize(y_true, dim=1)
    y_pred2 = torch.nn.functional.normalize(y_pred, dim=1)
    A = torch.mean(y_true2 * y_pred2)
    sad = torch.acos(A)
    return sad