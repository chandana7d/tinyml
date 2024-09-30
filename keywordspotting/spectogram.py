import os
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wav_read
import tensorflow_io as tfio
import librosa
import sounddevice as sd
import wave
import tensorflow as tf

"""
The AudioProcessor class is designed to streamline the process of recording, saving, 
and analyzing audio samples for various applications. With functionalities that include capturing audio in real-time, 
aving it in standard formats, and visualizing its characteristics through multiple plots such as 
1. signal waveforms
2. Fourier transforms
3. spectrograms
4. Mel-frequency cepstral coefficients (MFCCs)

Captures four distinct audio recordings (two "YES" samples and two "NO" samples) and saves each to separate WAV files. 
The recordings vary based on the loudness and the content being said, allowing for varied data collection for potential analysis or machine learning tasks.
"""
class AudioProcessor:
    def __init__(self, output_dir='output', sample_rate=16000):
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def record_audio(self, duration, type):
        print("Recording " + type + " audio...")
        audio_data = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        return np.squeeze(audio_data), self.sample_rate

    def save_audio(self, file_name, audio_data):
        file_path = os.path.join(self.output_dir, file_name)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio saved as {file_path}")

    def plot_signals(self, audios, titles):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        max_val = max([max(abs(audio)) for audio in audios])
        for i, ax in enumerate(axes.flatten()):
            ax.plot(audios[i])
            ax.set_title(titles[i], fontsize=20, fontweight='bold')
            ax.set_ylim(-max_val, max_val)
        fig.set_size_inches(18, 12)
        self._save_figure(fig, "signals.jpeg")
        plt.show()

    def plot_fourier_transforms(self, audios, titles):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        for i, ax in enumerate(axes.flatten()):
            ft_audio = np.abs(2 * np.fft.fft(audios[i]))
            ax.plot(ft_audio)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(titles[i], fontsize=20, fontweight='bold')
        fig.set_size_inches(18, 12)
        self._save_figure(fig, "fourier_transforms.jpeg")
        plt.show()

    def plot_spectrograms(self, audios, sample_rates, titles):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        for i, ax in enumerate(axes.flatten()):
            spectrogram = tfio.audio.spectrogram(audios[i]/1.0, nfft=2048, window=len(audios[i]), stride=int(sample_rates[i] * 0.008))
            ax.imshow(tf.math.log(spectrogram).numpy(), aspect='auto')
            ax.set_title(titles[i], fontsize=20, fontweight='bold')
        fig.set_size_inches(18, 12)
        self._save_figure(fig, "spectrograms.jpeg")
        plt.show()

    def plot_mfcc(self, audios, sample_rates, titles):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        for i, ax in enumerate(axes.flatten()):
            mfcc_data = librosa.power_to_db(librosa.feature.melspectrogram(y=np.float32(audios[i]), sr=sample_rates[i], n_fft=2048, hop_length=512, n_mels=128), ref=np.max)
            ax.imshow(np.swapaxes(mfcc_data, 0, 1), interpolation='nearest', cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(titles[i], fontsize=20, fontweight='bold')
            ax.set_ylim(ax.get_ylim()[::-1])
        fig.set_size_inches(18, 12)
        self._save_figure(fig, "mfcc.jpeg")
        plt.show()

    def _save_figure(self, fig, file_name):
        file_path = os.path.join(self.output_dir, file_name)
        fig.savefig(file_path, format='jpeg')
        print(f"Figure saved as {file_path}")


# Main Program

if __name__ == "__main__":
    # Instantiate the AudioProcessor class
    audio_processor = AudioProcessor()

    # Record different audio samples
    audio_yes_loud, sr_yes_loud = audio_processor.record_audio(3, "loud YES")  # Record for 3 seconds
    audio_processor.save_audio("audio_yes_loud.wav", audio_yes_loud)

    audio_yes_quiet, sr_yes_quiet = audio_processor.record_audio(3,"quiet YES")
    audio_processor.save_audio("audio_yes_quiet.wav", audio_yes_quiet)

    audio_no_loud, sr_no_loud = audio_processor.record_audio(3,"loud NO")
    audio_processor.save_audio("audio_no_loud.wav", audio_no_loud)

    audio_no_quiet, sr_no_quiet = audio_processor.record_audio(3, "quiet No")
    audio_processor.save_audio("audio_no_quiet.wav", audio_no_quiet)

    # Plot and save the visualizations
    audio_processor.plot_signals(
        [audio_yes_loud, audio_yes_quiet, audio_no_loud, audio_no_quiet],
        ["Yes Loud", "Yes Quiet", "No Loud", "No Quiet"]
    )

    audio_processor.plot_fourier_transforms(
        [audio_yes_loud, audio_yes_quiet, audio_no_loud, audio_no_quiet],
        ["Yes Loud", "Yes Quiet", "No Loud", "No Quiet"]
    )

    audio_processor.plot_spectrograms(
        [audio_yes_loud, audio_yes_quiet, audio_no_loud, audio_no_quiet],
        [sr_yes_loud, sr_yes_quiet, sr_no_loud, sr_no_quiet],
        ["Yes Loud", "Yes Quiet", "No Loud", "No Quiet"]
    )

    audio_processor.plot_mfcc(
        [audio_yes_loud, audio_yes_quiet, audio_no_loud, audio_no_quiet],
        [sr_yes_loud, sr_yes_quiet, sr_no_loud, sr_no_quiet],
        ["Yes Loud", "Yes Quiet", "No Loud", "No Quiet"]
    )
