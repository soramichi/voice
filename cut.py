import os
import wave

dir = "./data/ai"
#dir = "./data/sample"

# 1 sec
length_to_cut = 1

for f in os.listdir(dir):
    # skip directories
    if len(f.split(".")) == 1 or  f.split(".")[1] != "wav":
        continue
    
    w = wave.open(dir + "/" + f, "rb")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()
    length_this_file = nframes / framerate
    data = w.readframes(nframes)

    print(f, length_this_file)
    print(nchannels, sampwidth, framerate, nframes, comptype, compname)

    n_cut = int(length_this_file / length_to_cut)
    nbytes_per_file = (length_to_cut * nchannels * sampwidth * framerate)

    for i in range(0, n_cut):
        filename = ("%s/cut/%s_%d.wav" % (dir, f.split(".")[0], i+1))
        w_out = wave.open(filename, "wb")
        w_out.setnchannels(nchannels)
        w_out.setframerate(framerate)
        w_out.setsampwidth(sampwidth)
        w_out.writeframes(data[(i * nbytes_per_file) : ((i+1) * nbytes_per_file)])

        

