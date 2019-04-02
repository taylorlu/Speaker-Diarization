import pyaudio
import wave

class AudioPlayer:
    """
    Player implemented with PyAudio

    http://people.csail.mit.edu/hubert/pyaudio/

    Mac OS X:

      brew install portaudio
      pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz
    """
    def __init__(self, wav):
        self.p = pyaudio.PyAudio()
        self.pos = 0
        self.stream = None
        self._open(wav)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wf.readframes(frame_count)
        self.pos += frame_count
        return (data, pyaudio.paContinue)

    def _open(self, wav):
        self.wf = wave.open(wav, 'rb')
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels = self.wf.getnchannels(),
                rate = self.wf.getframerate(),
                output=True,
                stream_callback=self.callback)
        self.pause()

    def play(self):
        self.stream.start_stream()

    def pause(self):
        self.stream.stop_stream()

    def seek(self, seconds = 0.0):
        sec = seconds * self.wf.getframerate()
        self.pos = int(sec)
        self.wf.setpos(int(sec))

    def time(self):
        return float(self.pos)/self.wf.getframerate()

    def playing(self):
        return self.stream.is_active()

    def close(self):
        self.stream.close()
        self.wf.close()
        self.p.terminate()

