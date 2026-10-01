import numpy as np, soundfile as sf
from scipy.linalg import toeplitz
import sys; sys.path.insert(0, '.')
from conjugate_mdf import CONJUGATE_MDF

mic, sr = sf.read('audio/microphone.wav')
FFT, STEP, N_G = 512, 128, 64
nbin = FFT//2+1
np.random.seed(3)
cg = CONJUGATE_MDF(NCHAN=1, NBIN=nbin, N_G=N_G, hop=128, Nrxref=1)
n_frames = (len(mic)-FFT)//STEP + 1
for i in range(n_frames):
    s=i*STEP
    X = np.fft.rfft(mic[s:s+FFT]).reshape(-1,1)
    cg.apply(X, X)

autoR = cg.autoR[0]; rcross = cg.rcross[0]; w = cg.w[0]
print("autoR vs rcross max|diff|:", np.abs(autoR - rcross[:,:,0]).max())
print("autoR max abs:", np.abs(autoR).max(), " mean abs:", np.abs(autoR).mean())
# per-bin: does class w solve its own system? and what's the residual w vs e1?
bad = []
for ib in range(nbin):
    T = toeplitz(autoR[ib])
    w_direct = np.linalg.solve(T + 1e-12*np.eye(N_G), rcross[ib,:,0])
    e1 = np.zeros(N_G, dtype=complex); e1[0]=1
    err_direct = np.linalg.norm(w_direct - e1)
    err_class  = np.linalg.norm(w[ib,:,0] - e1)
    if err_direct > 0.1 or err_class > 0.5:
        bad.append((ib, err_direct, err_class, np.abs(autoR[ib,0])))
bad.sort(key=lambda t: -t[2])
print("bins with large error (ib, |w_direct-e1|, |w_class-e1|, r0):")
for row in bad[:10]: print("  ", row)
print("num bad bins:", len(bad), "of", nbin)
