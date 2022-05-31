import os
import sys
import pickle
import wfdb
import numpy as np
import biosppy.signals.tools as st

from tqdm import tqdm
from scipy.signal import medfilt
from concurrent.futures import ProcessPoolExecutor, as_completed
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter


base_dir = './data_set/'
save_dir = './pre_processing_data/'

names = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
         "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
         "b01", "b02", "b03", "b04", "b05",
         "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10",

         "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
         "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
         "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
         "x31", "x32", "x33", "x34", "x35"]

step = 5

hz = 100
sample = hz * 60

before = 2
after = 2
hr_min = 20
hr_max = 300

num_worker = 35 if os.cpu_count() > 35 else os.cpu_count() - 1


def worker(name, labels):

    x = []
    y = []
    groups = []

    signals = wfdb.rdrecord(base_dir + f'{name}', channels=[0]).p_signal[:, 0]
    for k in tqdm(range(len(labels)), desc=name, file=sys.stdout):
        if k < before or (k + 1 + after) > len(signals) / float(sample):
            continue
        signal = signals[int((k - before) * sample):int((k + 1 + after) * sample)]
        signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * hz),
                                        frequency=[3, 45], sampling_rate=hz)

        r_peaks, = hamilton_segmenter(signal, sampling_rate=hz)
        r_peaks, = correct_rpeaks(signal, rpeaks=r_peaks, sampling_rate=hz, tol=0.1)
        if len(r_peaks) / (1 + after + before) < 40 or \
                len(r_peaks) / (1 + after + before) > 200:
            continue

        rri_tm, rri_signal = r_peaks[1:] / float(hz), np.diff(r_peaks) / float(hz)
        rri_signal = medfilt(rri_signal, kernel_size=3)
        amp_tm, amp_signal = r_peaks / float(hz), signal[r_peaks]
        hr = 60 / rri_signal

        if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
            x.append([(rri_tm, rri_signal), (amp_tm, amp_signal)])
            y.append(0. if labels[k] == 'N' else 1.)
            groups.append(name)

    return x, y, groups


if __name__ == "__main__":
    assert os.path.exists(save_dir)

    for i in range(0, len(names), step):
        with ProcessPoolExecutor(max_workers=num_worker) as executor:
            task_list = []
            for j in range(i, i + step):
                label = wfdb.rdann(base_dir + names[j], extension='apn').symbol
                task_list.append(executor.submit(worker, names[j], label))

            for task in as_completed(task_list):
                file_name = task.result()[2][0]
                with open(save_dir + f'{file_name}.pkl', 'wb') as f:
                    pickle.dump(task.result(), f)

        print(f'\n{i}~{i + step - 1} ok!\n')
