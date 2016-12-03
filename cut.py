import sys
import os
import wave

def get_target_dirs(base_dir = "./data"):
    dirs = os.listdir(base_dir)
    targets = []

    for d in dirs:
        cut_dir = ("%s/%s/cut" % (base_dir, d))

        try:
            s = os.stat(cut_dir)
        except FileNotFoundError:
            os.mkdir(cut_dir)

        if len(os.listdir(cut_dir)) > 0:
            print("Warning: %s is not empty. Skip." % cut_dir)
        else:
            targets.append(("%s/%s" % (base_dir, d)))

    return targets

def main():
    # setting
    length_to_cut = 1 # 1 sec

    targets = get_target_dirs()

    for d in targets:
        print("cut files in %s" % d)
        n = 0
        
        for f in os.listdir(d):
            # skip directories
            if len(f.split(".")) == 1 or  f.split(".")[1] != "wav":
                continue

            w = wave.open(d + "/" + f, "rb")
            (nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()
            length_this_file = nframes / framerate
            data = w.readframes(nframes)

            # for debug
            # print(f, length_this_file)
            # print(nchannels, sampwidth, framerate, nframes, comptype, compname)

            n_cut = int(length_this_file / length_to_cut)
            nbytes_per_file = (length_to_cut * nchannels * sampwidth * framerate)

            for i in range(0, n_cut):
                n += 1
                filename = ("%s/cut/%s_%d.wav" % (d, f.split(".")[0], i+1))
                w_out = wave.open(filename, "wb")
                w_out.setnchannels(nchannels)
                w_out.setframerate(framerate)
                w_out.setsampwidth(sampwidth)
                w_out.writeframes(data[(i * nbytes_per_file) : ((i+1) * nbytes_per_file)])

        print("Done. Generated %d data files." % n)

if __name__ == "__main__":
    main()
