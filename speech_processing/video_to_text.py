import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import shutil
import subprocess
import pandas as pd
from pathlib import Path
#from moviepy.editor import VideoFileClip
from moviepy import VideoFileClip

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton, QFileDialog, QComboBox, QLineEdit, QFormLayout, QMessageBox, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

class Worker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(str)
    
    def __init__(self, file_path, model, language,output_folder):
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.language = language
        self.output_folder = output_folder

    def run(self):
        file_ext = Path(self.file_path).suffix.lower()

        if file_ext == '.mp4':
            self.convert_mp4_to_mp3(self.file_path)
            mp3_file = self.file_path.replace('.mp4', '.mp3')
            self.process_audio(mp3_file)
        elif file_ext in {'.mp3', '.wav'}:
            self.process_audio(self.file_path)

        self.xlsx_file = Path(self.output_folder) / (Path(self.file_path).stem + '.xlsx')
        self.finished.emit(str(self.xlsx_file))

    def convert_mp4_to_mp3(self, mp4_file):
        mp3_file = mp4_file.replace('.mp4', '.mp3')
        video = VideoFileClip(mp4_file)
        video.audio.write_audiofile(mp3_file)
        video.close()

    def process_audio(self, audio_file):
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)

        srt_file = temp_dir / (Path(audio_file).stem + '.srt')
        lrc_file = Path(self.output_folder) / (Path(audio_file).stem + '.lrc')

        if lrc_file.exists():
            return        
        #subprocess.run(f'whisper-ctranslate2 "{audio_file}" --model {self.model} --language {self.language} --vad_filter True --print_colors True ', shell=True, cwd=temp_dir)
        #subprocess.run(f'whisper "{audio_file}" --model {self.model} --language {self.language}  --device cpu', shell=True, cwd=temp_dir)
        subprocess.run(f'whisper-ctranslate2 "{audio_file}" --model {self.model} --language {self.language} --vad_filter True --print_colors True  --device cpu', shell=True, cwd=temp_dir)
        if srt_file.exists():
            subprocess.run(f'python srt_to_lrc.py "{srt_file}"', shell=True)
            lrc_file = temp_dir / (Path(audio_file).stem + '.lrc')
            if lrc_file.exists():
                final_lrc_file = Path(self.output_folder) / (Path(audio_file).stem + '.lrc')
                shutil.move(lrc_file, final_lrc_file)

                if final_lrc_file.exists():
                    txt_file = Path(self.output_folder) / (Path(audio_file).stem + '.txt')
                    with open(final_lrc_file, 'r', encoding='utf-8') as lrc, open(txt_file, 'w', encoding='utf-8') as txt:
                        txt.write(lrc.read())

                    self.create_excel(final_lrc_file, Path(self.output_folder) / (Path(audio_file).stem + '.xlsx'))
        shutil.rmtree(temp_dir)

    def create_excel(self, lrc_file, output_excel):
        data = {'Time': [], 'Speaker': [], 'Sentence': []}

        with open(lrc_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    parts = line.split(']')
                    if len(parts) > 1:
                        time = parts[0].strip('[')
                        content = parts[1].strip()
                        if ':' in content:
                            speaker, sentence = content.split(':', 1)
                            data['Time'].append(time)
                            data['Speaker'].append(speaker.strip())
                            data['Sentence'].append(sentence.strip())
        
        df = pd.DataFrame(data)
        df.to_excel(output_excel, index=False)

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Processor")

        layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.file_path = QLineEdit()
        self.file_path.setText("D:/Landmark_CNN_tf5 - Copy/au/demo/Text2emotion/test_video/Dracula 1931.mp4")
        form_layout.addRow("File path", self.file_path)

        self.output_path = QLineEdit()
        self.output_path.setText("output")
        form_layout.addRow("Out path", self.output_path)

        self.model_select = QComboBox()
        self.model_select.addItems(["tiny", "base", "small", "medium", "large"])
        form_layout.addRow("Model select", self.model_select)

        self.language_input = QLineEdit()
        self.language_input.setText("en")
        form_layout.addRow("Language", self.language_input)
        layout.addLayout(form_layout)

        self.run_emo_checkbox = QCheckBox("Run emotion analysis")
        form_layout.addRow("Run emotion analysis", self.run_emo_checkbox)        

        self.progress_label = QLabel("Ready to start...")
        self.progress_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.progress_label)

        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("Arial", 12))
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.select_file_button = QPushButton("Select file")
        self.select_file_button.setFont(QFont("Arial", 12))
        self.select_file_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_file_button)

        self.output_path_button = QPushButton("Select output")
        self.output_path_button.setFont(QFont("Arial", 12))
        self.output_path_button.clicked.connect(self.select_output)
        layout.addWidget(self.output_path_button)

        self.setLayout(layout)

    def select_file(self):
        default_path = os.path.dirname(self.file_path.text()) if self.file_path.text() else "E:/Data/Input"
        file, _ = QFileDialog.getOpenFileName(self, "Select Audio or Video File", default_path, "Audio/Video Files (*.mp4 *.mp3 *.wav)")
        if file:
            self.file_path.setText(file)

    def select_output(self):
        default_path = os.path.dirname(self.output_path.text()) if self.output_path.text() else "E:/Data/Output"
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", default_path)
        if folder:
            self.output_path.setText(folder)

    def start_processing(self):
        file_path = self.file_path.text()
        output_folder = self.output_path.text()
        model = self.model_select.currentText()
        language = self.language_input.text()
        if not file_path or not language:
            QMessageBox.warning(self, "Warn", "Please make sure you have entered the file path and language.")
            return

        self.start_button.setEnabled(False)
        self.select_file_button.setEnabled(False)
        self.output_path_button.setEnabled(False)

        self.worker = Worker(file_path, model, language, output_folder)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.worker.finished.connect(self.run_emotion_analysis)

    def worker_finished(self):
        self.progress_label.setText("Done!")
        self.start_button.setEnabled(True)
        self.select_file_button.setEnabled(True)        
        self.output_path_button.setEnabled(True)

    def run_emotion_analysis(self, xlsx_file):
        if self.run_emo_checkbox.isChecked():
            subprocess.run(['python', 'E:/Thesis/Text2emotion/Text2Emo_xlsx.py', str(xlsx_file)])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())