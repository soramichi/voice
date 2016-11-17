import pickle
import os
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import chainer.links as L
import pdb

def forward(model, x_data, y_data, train=True, f_test=None):
    x, t = Variable(x_data, volatile=not train), Variable(y_data, volatile=not train)
    h = F.relu(model.bn1(model.conv1(x)))
    h = F.relu(model.bn2(model.conv2(h)))
    h = F.relu(model.conv3(h))
    h = F.dropout(F.relu(model.fl4(h)),train=train)
    y = model.fl5(h)

    if train == False:
        for i in range(0, len(y.data)):
            print(f_test[i], "o" if np.argmax(y.data[i]) == t.data[i] else "x")

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def evaluate(model, x_test, y_test, f_test, batchsize):
    # evaluation
    sum_accuracy = 0
    sum_loss     = 0
    for i in range(0, len(x_test), batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        f_batch = f_test[i:i+batchsize]

        loss, acc = forward(model, x_batch, y_batch, False, f_batch)

        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print('test  mean loss=%f, accuracy=%f' % (sum_loss / len(x_test), sum_accuracy / len(x_test)))

def load_data(dir, N_train_per_file, N_test_per_file):
    batches = os.listdir(dir)
    data_train, data_test = [], []
    labels_train, labels_test = [], []
    f_train, f_test = [], []

    for n in range(0, len(batches)):
        # f must have at least (N_train + N_test) lines
        f = open(dir + "/" + batches[n], "r")
        label = n

        # for training
        for i in range(0, N_train_per_file):
            v = f.readline().strip().split(",")
            f_train.append(v[0])
            data_train.append(np.array(list(map(float, v[1:])), ndmin=2).astype(np.float32))
            labels_train += [label]

        # for test
        for i in range(0, N_test_per_file):
            v = f.readline().strip().split(",")
            f_test.append(v[0])
            data_test.append(np.array(list(map(float, v[1:])), ndmin=2).astype(np.float32))
            labels_test += [label]

    max_label = labels_train[-1]

    return np.array(data_train, ndmin=3), np.array(data_test, ndmin=3), np.array(labels_train).astype(np.int32), np.array(labels_test).astype(np.int32), f_train, f_test, max_label

def main():
    # global setting
    model_file = "./model.dat"
    reload_model = False
    n_epoch = 3
    batchsize = 10
    N_train_per_file = 90
    N_test_per_file = 20

    x_train, x_test, y_train, y_test, f_train, f_test, max_label = load_data("./data_batch", N_train_per_file, N_test_per_file)
    N_train = len(x_train)
    N_test = len(x_test)

    # setup model
    if reload_model == False:
        print("reload_model is off, learn the model from scratch")
        model = FunctionSet(conv1=L.ConvolutionND(1, 1, 3, 20),
                            bn1   = F.BatchNormalization(3),
                            conv2=L.ConvolutionND(1, 3, 5, 5, pad=1),
                            bn2   = F.BatchNormalization(5),
                            conv3=L.ConvolutionND(1, 5, 5, 5, pad=1),
                            fl4=F.Linear(99885, 256),
                            fl5=F.Linear(256, max_label+1))

    else:
        print("reload_model is on, reload a given model from existing file")
        f_in = open(model_file, "rb")
        model = pickle.load(f_in, encoding="bytes")

    optimizer = optimizers.Adam()
    optimizer.setup(model.collect_parameters())

    # training, executed when not reload mode
    if reload_model == False:
        for epoch in range(1, n_epoch+1):
            print('epoch', epoch)

            perm = np.random.permutation(N_train)
            sum_accuracy = 0
            sum_loss = 0

            for i in range(0, N_train, batchsize):
                print("%d/%d" % (i, N_train))
                
                x_batch = x_train[perm[i:i+batchsize]]
                y_batch = y_train[perm[i:i+batchsize]]

                optimizer.zero_grads()
                loss, acc = forward(model, x_batch, y_batch)
                loss.backward()
                optimizer.update()
                
                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
                
            print('train mean loss=%f, accuracy=%f' % (sum_loss / N_train, sum_accuracy / N_train))
            evaluate(model, x_test, y_test, f_test, batchsize)

        print("learning done. save the learnt model into a file")
        f_out = open(model_file, "wb")
        pickle.dump(model, f_out)
    else:
        evaluate(model, x_test, y_test, f_test, batchsize)

if __name__ == "__main__":
    main()
