# import numerical processing library
"""
    The objective of this exercise is that you analyse the code provided and
    make the link with the curse. You have to provide a short report that
    comments and analyse the results. You can use directly the results or adapt
    them to you needs.

"""

# import the numerical library
import numpy as np
# import signal processing library
import scipy.signal as sp
# import ploting library
import pylab as py
py.ion()
py.close('all')

# load the ecg signal
x = np.genfromtxt('respiration.dat')
# sampling frequency of the signal is 500 Hz
fs = 2
# generate correponding time vector
t = np.arange(len(x))/fs

""" 
    The signal is a measurement of the breathing obtained by inductance
    plethysmography.

    The objective is to estimate the breathing frequency.
"""

""" 
    The Hilbert transforms permits to estimate the instaneous amplitude and
    phase of a narrow band signal. 

    Q: Comment the figures.
    On the plots, we see the original breathing signal with all its oscillations, and on top of it the Hilbert envelope. While the raw signal goes up and down at each inhale and exhale, the envelope draws a smoother curve that follows the global amplitude variations over time. This representation makes the breathing dynamics clearer, as the envelope highlights the modulation of the signal rather than each individual oscillation.
    Q: Why the envelope does no follow the maxima of the signal
    The envelope represents the instantaneous amplitude derived from the analytic signal using the Hilbert transform. This analytic signal is complex: its real part corresponds to the original signal x(t), while its imaginary part is the quadrature component. Therefore, it is normal that the envelope is not strictly identical to the raw signal. Unlike simple maxima and minima, which are sensitive to noise and waveform asymmetry, the envelope offers a smoother and more robust measure of amplitude. For this reason, it does not perfectly align with every peak of the signal, but instead provides a clearer and more reliable view of how the amplitude evolves over time.
"""

# compute the analytical signal of x (Hilbert transform)
xa = sp.hilbert(x)

# plot the signal
py.figure(1, figsize=[5,5])
py.clf()
py.plot(t, x, label='breathing signal')
py.plot(t, np.abs(xa), label='envelop')
py.xlabel('time (s)')
py.ylabel('amplitude (a.u.)')
py.legend(loc='upper right')
py.title('Breathing signal')

"""
    The raw breathing signal does not fullfil the requirement of narrow band.
    The normal range of frequency for the breathing is within 0.1 to 0.25 Hz.
    The signal is first filtered for this interval.

    Q: Comment the figures
    After applying a band-pass filter to the signal, it fulfills the narrowband requirement and thus allows the estimation of the instantaneous amplitude. In the plot, we observe the raw breathing signal $x(t)$ together with its envelope $|hilbert(x)|$. As mentioned earlier, it is expected that the envelope does not perfectly follow every maximum; instead, it provides a smoother and less noisy representation of the amplitude. In the raw signal, we can clearly distinguish narrowband oscillations corresponding to each inspiration–expiration cycle, with a frequency of about 0.17 Hz, which falls within the normal respiratory range.
    Q: How is the estimation of the amplitude envelope.
    Because the signal was band-pass filtered in the range of 0.1–0.25 Hz, it satisfies the narrowband condition required for the Hilbert method to produce reliable results. With this preprocessing, the envelope, obtained as the modulus of the analytic signal, provides a smooth and robust estimate of how the breathing amplitude evolves over time. As we can see in the plot, this approach works well.

"""


# Analogic limit of the passband frequency
f_pass = np.array([0.1, 0.25])
# Analogic limit of the stopband frequency
f_stop = np.array([0, 0.6])
# Convertion into Nyquist frequency
f_pass_N = f_pass/fs*2
f_stop_N = f_stop/fs*2
# Max attenutation in passband (dB)
g_pass = 3
# Min attenuation in stopband (dB)
g_stop = 40
# Determine the order and the cutoff frequency of a butterworth filter
ord, wn = sp.buttord(f_pass_N, f_stop_N, g_pass, g_stop)
# Compute the coeffcients of the filter
b, a = sp.butter(ord, wn, btype='band')
# Filter the signal
x_bp = sp.filtfilt(b ,a, x)

# Compute the Hilbert transform.
xa = sp.hilbert(x_bp)

py.figure(2, figsize=[5,5])
py.clf()
py.plot(t, x_bp, label='filtered breathing signal')
py.plot(t, np.abs(xa), label='envelop')
py.xlabel('time (s)')
py.ylabel('amplitude (a.u.)')
py.title('Filtered breathing signal')
py.legend(loc='upper right')

"""
    The angle of the Hilbert transform gives the instaneous phase of the signal.

    Q: Comment the figure.
    Q: What is the role of the unwrap function

"""

# estimate the instantaneous phase from the Hilbert transform
phi_xa = np.angle(xa)
# phase is bounded between -pi and pi -> reconstruct continuous signal
phi_xa_unw = np.unwrap(phi_xa)

py.figure(3, figsize=[5, 8])
py.clf()
py.subplot(2,1,1)
py.plot(t, phi_xa)
py.xlabel('time (s)')
py.ylabel('instantaneous phase (rad)')
py.subplot(2,1,2)
py.plot(t, phi_xa_unw)
py.xlabel('time (s)')
py.ylabel('instantaneous phase unwrapped (rad)')

"""
    The time derivate of the instaneous phase is the instaneous frequency of the
    signal.

    Q: Comment the figure.
    Q: Compare the original waveform with the estimation of the breathing
       frequency

"""

# compute the derivative of the phase (angular frequency).
d_phi = np.diff(phi_xa_unw)
# convert angular frequency to frequency.
d_phi /= 2*np.pi
# convert digital frequency to analog frequency and in breathing per minute
# (bpm)
d_phi *= fs*60

py.figure(4, figsize=[5,8])
py.clf()
py.subplot(2,1,1)
py.plot(t, x, label='breathing signal')
py.xlabel('time (s)')
py.ylabel('amplitude (a.u.)')
py.title('Breathing signal')
py.subplot(2,1,2)
py.plot(t[1:], d_phi)
py.xlabel('time (s)')
py.ylabel('breathing rate (bpm)')






