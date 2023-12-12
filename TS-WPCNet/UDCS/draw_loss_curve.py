import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from pylab import *
import math

def draw_loss_curve(epoch_train_loss,epoch_val_loss,epoch_mse_psnr,epoch_huber_loss):
    # Set up the figure and axes
    # fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#0077C8', '#C70039', '#ff4500', '#008B8B']
    f,(ax3,ax) = plt.subplots(2,1,sharex=False)
    plt.subplots_adjust(wspace=0,hspace=0.08)

    ax.set_xlim(0, len(avg_loss))
    ax.set_ylim(0, 50)

    ax3.set_xlim(0, len(avg_loss))
    ax3.set_ylim(0, 50)

    ax.plot(epoch_train_loss, color=colors[0], label='Training Loss')
    ax.plot(epoch_val_loss, color=colors[1], label='Validation Loss',linestyle='--')

    ax3.xaxis.set_major_locator(plt.NullLocator())

    ax3.plot(mse_psnr, color=colors[2], label='Training PSNR')
    ax3.plot(huber_psnr, color=colors[3], label='Validation PSNR',linestyle='--')

    ax3.set_ylim(0.05, 32) 
    ax.set_ylim(0, 0.03) 

    ax3.grid(axis='both', linestyle='--') 
    ax.grid(axis='y', linestyle='--') 

    # ax3.legend(loc='lower right')  
    f.legend( loc='center right', bbox_to_anchor=(0.9, 0.6))

    plt.xlabel("Epoch",fontsize=12) 
    ax.set_ylabel("Loss",fontsize=12)  
    ax3.set_ylabel("PSNR(dB)",fontsize=12)  

    ax.spines['top'].set_visible(False) 
    ax.spines['bottom'].set_visible(True) 
    ax.spines['right'].set_visible(False) 

    ax3.spines['top'].set_visible(False)  
    ax3.spines['bottom'].set_visible(False)  
    ax3.spines['right'].set_visible(False)  

    # ax.tick_params(labeltop='off')
    ax3.set_title("Loss and PSNR for Training and Validation Data")


    d = 0.01  
    kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
    ax3.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal

    kwargs.update(transform=ax.transAxes, color='k')  # switch to the bottom axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal


    plt.savefig("/home/dell/jzy/lab315/jzy/PaperLearning/MyUbuntu/AE_Cluster/VGG_AE_Cluster/CGAN/output_cluster_results_2d/loss.pdf")
    plt.show()


if __name__ == '__main__':

    with open('train.log', 'r') as f:
        content = f.readlines()

    avg_loss = [0.2273]
    val_loss = [0.1889]
    mse_psnr = [5.0]
    huber_psnr = [5.0]
    best_val_loss = [0.0]

    for line in content:
        if 'The average loss is' in line:
            avg_loss.append(float(line.split('is')[1]))
        elif 'Validation loss:' in line:
            val_loss.append(float(line.split(':')[2].split(',')[0]))
            mse_psnr.append(float(line.split(':')[3].split(',')[0]))
            a = line.split(':')[3].split('：')
            huber_psnr.append(float(line.split(':')[3].split('：')[-1]))
        elif 'The best validation loss is' in line:
            best_val_loss.append(float(line.split('is')[1].split('in')[0]))

    train_psnr = [20 * math.log10(1/math.sqrt(x)) for x in avg_loss]


    draw_loss_curve(avg_loss,val_loss,mse_psnr,train_psnr)

    # print("Average loss: ", avg_loss)
    # print("Validation loss: ", val_loss)
    # print("MSE_PSNR: ", mse_psnr)
    # print("Huber_PSNR: ", huber_psnr)
    # print("Best validation loss: ", best_val_loss)

