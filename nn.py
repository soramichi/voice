import pickle
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import chainer.links as L
import pdb

def forward(model, x_data, y_data, train=True, filename=None):
    x, t = Variable(x_data, volatile=not train), Variable(y_data, volatile=not train)
    h = F.relu(model.bn1(model.conv1(x)))
    h = F.relu(model.bn2(model.conv2(h)))
    h = F.relu(model.conv3(h))
    h = F.dropout(F.relu(model.fl4(h)),train=train)
    y = model.fl5(h)

    if train == False:
        print(filename, F.accuracy(y, t).data)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def load_data(filename, N):
    f = open(filename, "r")
    data = []
    filenames = []

    for i in range(0, N):
        v = f.readline().strip().split(",")
        filenames.append(v[0])
        data.append(np.array(list(map(float, v[1:])), ndmin=2).astype(np.float32))

    return data, filenames

def main():
    # global setting
    model_file = "./model.dat"
    reload_model = False
    n_epoch = 3
    batchsize = 10
    #N_train = 180
    #N_test = 20
    N_train = 90
    N_test = 20
    
    data_ai, files_ai = load_data("./data/ai.batch", N_train + N_test)
    data_kugi, files_kugi  = load_data("./data/kugi.batch", N_train + N_test)
    x_train = np.array(data_kugi[0:N_train] + data_ai[0:N_train])
    y_train = np.array([0] * N_train + [1] * N_train).astype(np.int32) # kugi:0, ai:1
    files_train = files_kugi[0:N_train] + files_ai[0:N_train]
    x_test = np.array(data_kugi[N_train:] + data_ai[N_train:])
    y_test = np.array([0] * N_test + [1] * N_test).astype(np.int32)
    files_test = files_kugi[N_train:] + files_ai[N_train:]
    
    # setup model
    if reload_model == False:
        print("reload_model is off, learn the model from scratch")
        model = FunctionSet(conv1=L.ConvolutionND(1, 1, 3, 20),
                            bn1   = F.BatchNormalization(3),
                            conv2=L.ConvolutionND(1, 3, 5, 5, pad=1),
                            bn2   = F.BatchNormalization(5),
                            conv3=L.ConvolutionND(1, 5, 5, 5, pad=1),
                            fl4=F.Linear(99885, 256),
                            fl5=F.Linear(256, 2))

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

            perm = np.random.permutation(2 * N_train)
            sum_accuracy = 0
            sum_loss = 0

            for i in range(0, 2 * N_train, batchsize):
                print("%d/%d" % (i, 2 * N_train))
                
                x_batch = x_train[perm[i:i+batchsize]]
                y_batch = y_train[perm[i:i+batchsize]]

                optimizer.zero_grads()
                loss, acc = forward(model, x_batch, y_batch)
                loss.backward()
                optimizer.update()
                
                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
                
            print('train mean loss=%f, accuracy=%f' % (sum_loss / (2 * N_train), sum_accuracy / (2 * N_train)))

        print("learning done. save the learnt model into a file")
        f_out = open(model_file, "wb")
        pickle.dump(model, f_out)
    
    # evaluation
    sum_accuracy = 0
    sum_loss     = 0
    batchsize = 1
    for i in range(0, 2 * N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]

        loss, acc = forward(model, x_batch, y_batch, train=False, filename=files_test[i])

        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print('test  mean loss=%f, accuracy=%f' % (sum_loss / (2 * N_test), sum_accuracy / (2 * N_test)))

if __name__ == "__main__":
    main()
