import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 1000  # Hz
duration = 10  # seconds
time = np.linspace(0, duration, sampling_rate * duration)

# Baseline EMG signal (Gaussian noise)
baseline_emg = np.random.normal(0, 0.5, size=time.shape)

# Tremor characteristics
tremor_frequency = 5  # Hz (typical for Parkinson's tremor)
tremor_amplitude = 1.0

# Simulate tremor as a sinusoidal signal
tremor_signal = tremor_amplitude * np.sin(2 * np.pi * tremor_frequency * time)

# Combine baseline EMG with tremor
simulated_emg = baseline_emg + tremor_signal

# Plot the signals
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title('Baseline EMG Signal')
plt.plot(time, baseline_emg)
plt.subplot(3, 1, 2)
plt.title('Tremor Signal')
plt.plot(time, tremor_signal, color='red')
plt.subplot(3, 1, 3)
plt.title('Simulated EMG Signal with Tremor')
plt.plot(time, simulated_emg, color='green')
plt.tight_layout()
plt.show()
