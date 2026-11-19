# Synthesises the launch sounds (needs only numpy; Blender's Python has it):
#   blender -b --python tools/rocket/make_sounds.py
# Output (assets/sounds/):
#   rocket_roar.wav  8 s of engine roar under a Hann envelope - the game overlaps two voices
#                    four seconds apart (as it does with the rain), so the sum is steady
#   beep.wav         the count's tick
import os, wave
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'sounds')
SR = 44100


def save(name, x):
    x = np.clip(x / (np.max(np.abs(x)) + 1e-9) * 0.92, -1, 1)
    with wave.open(os.path.join(OUT, name), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((x * 32767.0).astype('<i2').tobytes())


def band(noise, lo, hi):
    """Band-limit white noise in the frequency domain."""
    X = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(len(noise), 1.0 / SR)
    g = 1.0 / (1.0 + (lo / np.maximum(fr, 1e-3)) ** 4) / (1.0 + (fr / hi) ** 4)
    return np.fft.irfft(X * g, len(noise))


rng = np.random.default_rng(3)
n = SR * 8
t = np.arange(n) / SR
w = rng.standard_normal(n)
rumble = band(w, 18.0, 90.0) * 3.2                    # the ground-shaking bottom
body = band(rng.standard_normal(n), 60.0, 450.0) * 1.3
hiss = band(rng.standard_normal(n), 400.0, 2600.0) * 0.30
# Crackle: the roar of a big rocket is torn by shock noise - short bursts, a few tens a second.
env = band(rng.standard_normal(n), 6.0, 45.0)
env = np.clip(env / (np.std(env) + 1e-9), -3, 3)
crackle = band(rng.standard_normal(n), 250.0, 5000.0) * np.maximum(env - 0.6, 0.0) ** 2 * 1.4
slow = 1.0 + 0.18 * band(rng.standard_normal(n), 0.3, 3.0) / 0.02 * 0.02
x = (rumble + body + hiss + crackle) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.37 * t + 1.0)) * slow
x = np.tanh(x * 0.9)                                  # a little saturation, as loud things have
x *= 0.5 - 0.5 * np.cos(2 * np.pi * t / 8.0)          # Hann: two voices 4 s apart add up to a constant
save('rocket_roar.wav', x)

nb = int(SR * 0.16)
tb = np.arange(nb) / SR
b = np.sin(2 * np.pi * 880.0 * tb) * 0.7 + np.sin(2 * np.pi * 1760.0 * tb) * 0.2
b *= np.minimum(1.0, tb / 0.004) * np.exp(-tb / 0.06)
save('beep.wav', b)
print('SOUNDS OK')
