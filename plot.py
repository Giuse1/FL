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
train_loss0, train_acc0, val_loss0, val_acc0 = read_file("results/real_CNNMnist_f_50r_1le_1u_8b_0.01lr.txt")
train_loss1, train_acc1, val_loss1, val_acc1 = read_file("results/real_CNNMnist_f_50r_1le_10u_8b_0.01lr.txt")
train_loss2, train_acc2, val_loss2, val_acc2 = read_file("results/real_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
train_loss3, train_acc3, val_loss3, val_acc3 = read_file("results/real_CNNMnist_f_50r_1le_1000u_8b_0.01lr.txt")

plt.figure()
plt.plot(val_loss0, label='1 client per round')
plt.plot(val_loss1, label='10 clients per round')
plt.plot(val_loss2, label='150 clients per round')
plt.plot(val_loss3, label='1000 clients per round')
plt.legend()
plt.title("Test loss w.r.t. number of clients")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.xticks(range(0,50,5))
plt.savefig("plots/Test loss w.r.t. number of clients.png")

plt.figure()
plt.plot(val_acc0, label='1 client per round')
plt.plot(val_acc1, label='10 clients per round')
plt.plot(val_acc2, label='150 clients per round')
plt.plot(val_acc3, label='1000 clients per round')
plt.legend()
plt.title("Test accuracy w.r.t. number of clients")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.xticks(range(0,50,5))
plt.yticks(np.arange(0,1,0.1))
plt.savefig("plots/Test accuracy w.r.t. number of clients.png")

train_loss1, train_acc1, val_loss1, val_acc1 = read_file("results/first150_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
train_loss2, train_acc2, val_loss2, val_acc2 = read_file("results/first150group2_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
train_loss3, train_acc3, val_loss3, val_acc3 = read_file("results/first150group3_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
train_loss4, train_acc4, val_loss4, val_acc4 = read_file("results/first150group5_CNNMnist_f_50r_1le_150u_8b_0.01lr.txt")
train_loss5, train_acc5, val_loss5, val_acc5 = read_file("results/hybrid_fl_150u_g2.txt")
train_loss6, train_acc6, val_loss6, val_acc6 = read_file("results/hybrid_fl_150u_g5.txt")
train_loss7, train_acc7, val_loss7, val_acc7 = read_file("results/hybrid_random_fl_150u_g2.txt")
train_loss8, train_acc8, val_loss8, val_acc8 = read_file("results/first150_CNNMnist_f_50r_2le_150u_8b_0.01lr.txt")
train_loss9, train_acc9, val_loss9, val_acc9 = read_file("results/hybrid_fl_150u_2le_g2.txt")




plt.figure()
plt.plot(val_loss1, label='150 users')
plt.plot(val_loss2, label='75 users')
plt.plot(val_loss3, label='50 users')
plt.plot(val_loss4, label='30 users')
#plt.plot(val_loss5,  label='150 users grouped in 2')

plt.legend()
plt.title("Test loss with same amount of training samples")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.xticks(range(0,50,5))
plt.savefig("plots/Test loss with same amount of training samples.png")

plt.figure()
plt.plot(val_acc1, label='Standard FL - 150 users')
#plt.plot(val_acc2, color="tab:orange", label='Standard FL - 75 users')
#plt.plot(val_acc3, label='Standard FL - 50 users')
plt.plot(val_acc4, color="tab:red", label='Standard FL - 30 users')
#plt.plot(val_acc5, '.-', color="tab:orange", label='Hybrid FL - 150 users with groups of 2')
#plt.plot(val_acc7, '-..',color="tab:blue", label='Hybrid random FL - 150 users with groups of 2 ')
plt.plot(val_acc6, '.-', color='tab:red', label='Hybrid FL - 150 users with groups of 5')
# plt.plot(val_acc8, color="tab:green", label='150 users with 2 local epochs')
# plt.plot(val_acc9, '.-', color="tab:green", label='HFL - 150 users with groups of 2 with 2 epochs')



plt.grid()
plt.legend()
plt.title("Test accuracy with same amount of training samples")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(range(0,50,5))
plt.yticks(np.arange(0,1,0.1))
plt.savefig("plots/Test accuracy with same amount of training samples.png")

