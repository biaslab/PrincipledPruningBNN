# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



# Custom function for plotting losses after training
def PlotTrainingLosses(kl_theta, kl_tau, acc_loss, figsize=[12,8]):
    """
    This function plots the VFE loss and its sperates terms.
    """
    
    # Generate gridspec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    # Create epochs
    n = np.arange(1, len(kl_theta)+1)

    # Change font-sizes
    fs_t = 18
    fs_x = 16
    fs_l = 14

    # Plot full loss
    ax1.plot(n, kl_theta+kl_tau+acc_loss)
    ax1.set_title("Variational Free Energy", fontsize=fs_t)
    ax1.set_xlabel('epoch', fontsize=fs_x)
    ax1.set_ylabel('loss', fontsize=fs_x)
    ax1.grid(linewidth=0.5, alpha=0.5)

    # Plot data-term
    ax2.plot(n, acc_loss)
    ax2.set_title("Accuracy Loss", fontsize=fs_t)
    ax2.set_xlabel('epoch', fontsize=fs_x)
    ax2.set_ylabel('loss', fontsize=fs_x)
    ax2.grid(linewidth=0.5, alpha=0.5)

    # Plot KL-term
    ax3.plot(n, kl_theta+kl_tau)
    ax3.set_title("Complexity Loss", fontsize=fs_t)
    ax3.set_xlabel('epoch', fontsize=fs_x)
    ax3.set_ylabel('loss', fontsize=fs_x)
    ax3.grid(linewidth=0.5, alpha=0.5)

    # Plot Theta KL-div.
    ax4.plot(n, kl_theta)
    ax4.set_title("Seperate KL-div.", fontsize=fs_t)
    ax4.set_xlabel('epoch', fontsize=fs_x)
    ax4.set_ylabel('theta', fontsize=fs_x, color='C0')
    ax4.tick_params(axis='y', labelcolor='C0')
    # Plot Tau KL-div.
    ax5 = ax4.twinx()
    ax5.plot(n, kl_tau, color='C1')
    ax5.set_ylabel('tau', fontsize=fs_x, color='C1')
    ax5.tick_params(axis='y', labelcolor='C1')
    ax4.grid(linewidth=0.5, alpha=0.5);