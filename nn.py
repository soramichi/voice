import pickle
import os
import numpy as np
from chainer import cuda, Chain, Variable, FunctionSet, optimizers, datasets, iterators, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

class MLP(Chain):
    def __init__(self, n_out):
        super(MLP, self).__init__(
            conv1=L.ConvolutionND(1, 1, 3, 20),
            bn1   = F.BatchNormalization(3),
            conv2=L.ConvolutionND(1, 3, 5, 5, pad=1),
            bn2   = F.BatchNormalization(5),
            conv3=L.ConvolutionND(1, 5, 5, 5, pad=1),
            fl4=F.Linear(149885, 256),
            fl5=F.Linear(256, n_out)
        )
    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.conv3(h))
        h = F.dropout(F.relu(self.fl4(h)), train=self.train)
        y = self.fl5(h)
        return y
    def enable_layer_output(self):
        self.output = True
    def disable_layer_output(self):
        self.output = False

def print_report(model, x_test, y_test, f_test, batchsize):
    f_out = open("./report.txt", "w")
    n_correct = 0

    for i in range(0, len(x_test), batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        f_batch = f_test[i:i+batchsize]

        t = model.predictor(x_batch)
        for i in range(0, batchsize):
            correct = (y_batch[i] == np.argmax(t.data[i]))
            n_correct += (1 if correct else 0)
            f_out.write("%s %c\n" % (f_batch[i], "o" if correct else "x"))

    f_out.write("Accuracy: %d/%d = %f\n" % (n_correct, len(x_test), n_correct / len(x_test)))

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

def make_tuple_dataset(x, y):
    tuples = []
    for i in range(0, len(x)):
        tuples.append((x[i], y[i]))
    return tuples
    
def main():
    # global setting
    model_file = "./model.dat"
    reload_model = False
    n_epoch = 5
    batchsize = 20
    N_train_per_file = 180
    N_test_per_file = 20

    x_train, x_test, y_train, y_test, f_train, f_test, max_label = load_data("./data_batch", N_train_per_file, N_test_per_file)
    N_train = len(x_train)
    N_test = len(x_test)

    # setup model
    if reload_model == False:
        print("reload_model is off, learn the model from scratch")
        model = L.Classifier(MLP(max_label + 1))
        model.predictor.train = True
    else:
        print("reload_model is on, reload a given model from existing file")
        f_in = open(model_file, "rb")
        model = pickle.load(f_in, encoding="bytes")

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train = make_tuple_dataset(x_train, y_train)
    test  = make_tuple_dataset(x_test, y_test)
    
    # training, executed when not reload mode
    if reload_model == False:
        train_iter = iterators.SerialIterator(train, batchsize)
        test_iter = iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)

        # Set up a trainer
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (n_epoch, 'epoch'))

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, model))
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport(log_name=None))
        
        # Print a progress bar
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy']))
        trainer.extend(extensions.ProgressBar(update_interval=1))

        # Run the training
        trainer.run()

        print("learning done. save the learnt model into a file")
        f_out = open(model_file, "wb")
        pickle.dump(model, f_out)

        print_report(model, x_test, y_test, f_test, batchsize)
    else:
        evaluate(model, x_test, y_test, f_test, batchsize)

if __name__ == "__main__":
    main()
