import subprocess
import sys


class MidiPlayer:
    def __init__(self, soundfont_path=None):
        self.soundfont_path = soundfont_path
        self.player_process = None

    def play_midi(self, filename):
        if self.player_process:
            self.stop_midi()

        cmd = ["fluidsynth", "-ni"]
        if self.soundfont_path:
            cmd.append(self.soundfont_path)
        cmd.append(filename)

        self.player_process = subprocess.Popen(cmd)

    def stop_midi(self):
        if not self.player_process:
            return

        if sys.platform.startswith("win"):
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.player_process.pid)])
        else:
            self.player_process.kill()

        self.player_process = None
