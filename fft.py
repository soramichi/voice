import os
import sys
import wave
import pickle
import functools
import numpy as np
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
"""

base_dir = sys.argv[1]
dir = base_dir + "/cut"

filelist = os.listdir(dir)

for i in range(0, len(filelist)):
    f = filelist[i]
    
    sys.stderr.write("%s (%d/%d)\n" % (f, i+1, len(filelist)))
    
    w = wave.open("%s/%s" % (dir, f), "rb")
    fs = w.getframerate()

    x = w.readframes(w.getnframes())
    x = np.frombuffer(x, dtype= "int16") / 32768.0

    X = np.fft.fft(x)
    data = functools.reduce(lambda l,c: np.append(l, np.array(c.real, c.imag).astype(np.float16)), X, np.array([]))

    # save only up to 15kHz
    # Note: one Hz has two data elements (real and imaginary parts)
    sys.stdout.write("%s, %s\n" % (dir + "/" + f, list(data[0:30000]).__str__()[1:-1]))

"""
plt.plot(freqList, amplitudeSpectrum, marker= 'o', linestyle='-')
plt.axis([0, 1600, 0, 250])
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude spectrum")
plt.savefig("test.png")
"""
