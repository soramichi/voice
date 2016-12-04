import os
import sys
import wave
import pickle
import functools
from multiprocessing import Pool
import numpy as np

output_dir = "./data_batch"

def get_target_dirs(base_dir = "./data"):
    dirs = os.listdir(base_dir)
    targets = []

    for d in dirs:
        cut_dir = ("%s/%s/cut" % (base_dir, d))

        try:
            s = os.stat(cut_dir)
        except FileNotFoundError:
            print("Error: %s does not include 'cut' directory" % d)
            sys.exit(1)

        if len(os.listdir(cut_dir)) == 0:
            print("Error: %s does not include data files." % cut_dir)
        else:
            targets.append("%s/%s" % (base_dir, d))

    return targets

def fft(target_dir):
    basename = target_dir.split("/")[-1]
    target_dir += "/cut"
    filelist = os.listdir(target_dir)

    print(target_dir, output_dir)
    print(basename)

    f_out = open("%s/%s.batch" % (output_dir, basename), "w")

    for i in range(0, len(filelist)):
        f = filelist[i]
    
        sys.stderr.write("%s (%d/%d)\n" % (f, i+1, len(filelist)))

        w = wave.open("%s/%s" % (target_dir, f), "rb")
        fs = w.getframerate()

        x = w.readframes(w.getnframes())
        x = np.frombuffer(x, dtype= "int16") / 32768.0

        X = np.fft.fft(x)
        data = functools.reduce(lambda l,c: np.append(l, np.array([c.real, c.imag]).astype(np.float16)), X, np.array([]))

        # save only up to 15kHz
        # Note: one Hz has two data elements (real and imaginary parts)
        f_out.write("%s, %s\n" % (target_dir + "/" + f, list(data[0:30000]).__str__()[1:-1]))
        f_out.flush()

def main():
    try:
        s = os.stat(output_dir)
    except FileNotFoundError:
        os.mkdir(output_dir)

    target_dirs = get_target_dirs()
    p = Pool(os.cpu_count())
    p.map(fft, target_dirs)

if __name__ == "__main__":
    main()
