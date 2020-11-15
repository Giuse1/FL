import matplotlib.pyplot as plt
import numpy as np


def read_file(path):
    f = open(path, "r")
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for line in f:
        a = line.split(" ")
        if a[0] == "train":
            train_loss.append(float(a[2]))
            train_acc.append(float(a[4]))
        elif a[0] == "val":
            val_loss.append(float(a[2]))
            val_acc.append(float(a[4]))

    return train_loss, train_acc, val_loss, val_acc




train_loss0, train_acc0, val_loss0, val_acc0 = read_file("results_weighted/hybrid_nonIID_random_150u_1le")
train_loss1, train_acc1, val_loss1, val_acc1 = read_file("results_weighted/hybrid_nonIID_random_150u_1le_complementary")
train_loss7, train_acc7, val_loss7, val_acc7 = read_file("results_weighted/hybrid_nonIID_random_150u_1le_seed1")
_, _, _, val_acc2 = read_file("results_weighted/hybrid_nonIID_random_150u_1le_same")


plt.figure()
plt.plot(val_acc0, label='HFL - Random selection (seed 0)')
plt.plot(val_acc7, label='HFL - Random selection (seed 1)')
plt.plot(val_acc1, label='HFL - Complementary users')
plt.plot(val_acc2, label='HFL - Same kind of users')


plt.legend()
plt.grid()
#plt.xticks(range(0, 101, 5))

plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Accuracy w.r.t. to number of rounds with non-IID data")
plt.savefig("non_iid comparison different scenarios")
#plt.show()

#
# train_loss0, train_acc0, val_loss0, val_acc0 = read_file("results_weighted/standard_nonIID_150u_1le")
# train_loss1, train_acc1, val_loss1, val_acc1 = read_file("results_weighted/standard_nonIID_75u_1le")
# plt.figure()
# plt.plot( val_acc0, label='SFL - 1 local epoch, 150 users')
# plt.plot( val_acc7, label='SFL - 1 local epoch, 75 users')
# plt.legend()
# plt.grid()
# #plt.xticks(range(0, 101, 5))
#
# plt.yticks(np.arange(0, 1, 0.1))
# plt.xlabel("Number of transmissions")
# plt.ylabel("Accuracy")
# plt.title("Accuracy w.r.t. to number of transmissions")
# plt.show()

# train_loss0, train_acc0, val_loss0, val_acc0 = read_file("results_weighted/standard_nonIID_first150u_1le")
# train_loss1, train_acc1, val_loss1, val_acc1 = read_file("results_weighted/hybrid_nonIID_first150u_1le")
# train_loss6, train_acc6, val_loss6, val_acc6 = read_file("results_weighted/hybrid_nonIID_random_first150u_1le")
# train_loss7, train_acc7, val_loss7, val_acc7 = read_file("results_weighted/hybrid_nonIID_random_first150_75u_1le")
#
# plt.figure()
# plt.plot(np.arange(0,25,0.5), val_acc0, label='SFL - 1 local epoch')
# plt.plot(val_acc1,  label='HFL - 1 local epoch, groups of 2')
# plt.plot(val_acc6,  label='HFL - 1 local epoch, groups of 5')
# plt.plot(np.arange(0,50,0.5), val_acc7, color="tab:red", label='HFL - 1 dewdewfwe epoch, groups of 5')
#
# plt.legend()
# plt.grid()
# plt.xticks(range(0, 50, 5))
#
# plt.yticks(np.arange(0, 1, 0.1))
# plt.show()
# plt.plot(val_acc2, color="tab:orange", label='SFL - 2 local epochs')
# plt.plot(val_acc3, '.-', color="tab:orange", label='HFL - 2 local epochs, groups of 2')
# plt.plot(val_acc7, 'x-', color="tab:orange", label='HFL - 2 local epochs, groups of 5')
# plt.plot(val_acc4, color="tab:green", label='SFL - 5 local epochs')
# plt.plot(val_acc5, '.-', color="tab:green", label='HFL - 5 local epochs, groups of 2')
# plt.plot(val_acc8, 'x-', color="tab:green", label='HFL - 5 local epochs, groups of 5')
#
# plt.legend()
#
# plt.title("Test accuracy with 150 randomly chosen users - SFL vs HFL")
# plt.xlabel("Round")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.xticks(range(0, 50, 5))
# plt.yticks(np.arange(0, 1, 0.1))
# #plt.savefig("plots/Test accuracy SFL vs HFL 2 epochs.png")
# plt.show()
#
# # plt.figure()
# # plt.plot(val_acc0, color="tab:blue", label='SFL - 1 local epoch')
# # plt.plot(val_acc1, '.-', color="tab:blue", label='HFL - 1 local epoch, groups of 2')
# plt.plot(val_acc6, 'x-', color="tab:blue", label='HFL - 1 local epoch, groups of 5')
# plt.plot(val_acc2, color="tab:orange", label='SFL - 2 local epochs')
# plt.plot(val_acc3, '.-', color="tab:orange", label='HFL - 2 local epochs, groups of 2')
# plt.plot(val_acc7, 'x-', color="tab:orange", label='HFL - 2 local epochs, groups of 5')
# plt.plot(val_acc4, color="tab:green", label='SFL - 5 local epochs')
# plt.plot(val_acc5, '.-', color="tab:green", label='HFL - 5 local epochs, groups of 2')
# plt.plot(val_acc8, 'x-', color="tab:green", label='HFL - 5 local epochs, groups of 5')
# plt.legend(loc=2, prop={'size': 10})
#
# plt.title("Test accuracy with 150 randomly chosen users - SFL vs HFL")
# plt.xlabel("Round")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.xticks(range(0, 50, 5))
# plt.yticks(np.arange(0, 1.01, 0.01))
# plt.ylim((0.9, 1.05))
# plt.savefig("plots/zoom test accuracy SFL vs HFL 2 epochs.png")
#
# #
# train_loss1, train_acc1, val_loss1, val_acc1 = read_file("results/first150_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
# train_loss2, train_acc2, val_loss2, val_acc2 = read_file("results/first150group2_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
# train_loss3, train_acc3, val_loss3, val_acc3 = read_file("results/first150group3_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
# train_loss4, train_acc4, val_loss4, val_acc4 = read_file("results/first150group5_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
# train_loss5, train_acc5, val_loss5, val_acc5 = read_file("results/hybrid_fl_150u_g2.txt")
# train_loss6, train_acc6, val_loss6, val_acc6 = read_file("results/hybrid_fl_150u_g5.txt")
# train_loss7, train_acc7, val_loss7, val_acc7 = read_file("results/hybrid_random_fl_150u_g2.txt")
# train_loss8, train_acc8, val_loss8, val_acc8 = read_file("results/first150_CNNMnist_f_50r_2le_150u_8b_0.01lr.txt")
# train_loss9, train_acc9, val_loss9, val_acc9 = read_file("results/hybrid_fl_150u_2le_g2.txt")
# train_loss10, train_acc10, val_loss10, val_acc10 = read_file("results/hybrid_fl_150u_5le_g5.txt")
# train_loss11, train_acc11, val_loss11, val_acc11 = read_file("results/first150_CNNMnist_f_50r_5le_150u_8b_0.01lr.txt")
#
#
#
#
#
# plt.figure()
# plt.plot(val_loss1, label='150 users')
# plt.plot(val_loss2, label='75 users')
# plt.plot(val_loss3, label='50 users')
# plt.plot(val_loss4, label='30 users')
# #plt.plot(val_loss5,  label='150 users grouped in 2')
#
# plt.legend()
# plt.title("Test loss with same amount of training samples")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid()
# plt.xticks(range(0,50,5))
# plt.savefig("plots/Test loss with same amount of training samples.png")
#
# plt.figure()
# plt.plot(val_acc1, label='Standard FL - 150 users')
# #plt.plot(val_acc2, color="tab:orange", label='Standard FL - 75 users')
# #plt.plot(val_acc3, label='Standard FL - 50 users')
# plt.plot(val_acc4, color="tab:red", label='Standard FL - 30 users')
# #plt.plot(val_acc5, '.-', color="tab:orange", label='Hybrid FL - 150 users with groups of 2')
# #plt.plot(val_acc7, '-..',color="tab:blue", label='Hybrid random FL - 150 users with groups of 2 ')
# plt.plot(val_acc6, '.-', color='tab:blue', label='Hybrid FL - 150 users with groups of 5')
# plt.plot(val_acc10, '.-', color='tab:green', label='Hybrid FL - 150 users with groups of 5, 5 local epochs')
# plt.plot(val_acc11, color='tab:green', label='Standard FL - 150 users with 5 local epochs')
#
# # plt.plot(val_acc8, color="tab:green", label='150 users with 2 local epochs')
# # plt.plot(val_acc9, '.-', color="tab:green", label='HFL - 150 users with groups of 2 with 2 epochs')
#
#
#
# plt.grid()
# plt.legend()
# plt.title("Test accuracy with same amount of training samples")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.xticks(range(0,50,5))
# plt.yticks(np.arange(0,1,0.1))
# plt.savefig("plots/Test accuracy with same amount of training samples.png")
