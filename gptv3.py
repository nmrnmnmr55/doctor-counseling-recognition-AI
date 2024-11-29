import streamlit as st
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
import os
import threading
import json
import queue
import time
import logging
import pandas as pd

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('app_debug.log', encoding='utf-8', mode='w')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
def replace_text(text, replacements):
    for old, new in replacements.items():
        if new is None:
            new = ""
        text = text.replace(old, str(new))
    return text
# Read replacements from an Excel file
replacements_df = pd.read_excel('replacements.xlsx')
replacements = dict(zip(replacements_df['old'], replacements_df['new']))

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TRANSCRIPTION_FILE = "transcriptions.json"

class ThreadSafeState:
    def __init__(self):
        self._lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.recording = False
        self.transcriptions = []
        self.summary = ""

    def set_recording(self, value):
        with self._lock:
            self.recording = value

    def get_recording(self):
        with self._lock:
            return self.recording

    def add_transcription(self, transcription):
        with self._lock:
            self.transcriptions.append(transcription)

    def get_transcriptions(self):
        with self._lock:
            return self.transcriptions.copy()

    def clear_transcriptions(self):
        with self._lock:
            self.transcriptions.clear()

    def set_summary(self, summary):
        with self._lock:
            self.summary = summary

    def get_summary(self):
        with self._lock:
            return self.summary

    def clear_summary(self):
        with self._lock:
            self.summary = ""

@st.cache_resource
def get_global_state():
    return ThreadSafeState()

global_state = get_global_state()

def transcribe_audio(audio_data):
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)
        
        with open("temp_audio.wav", "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="これは医療相談の音声です。患者と医師の会話を正確に文字起こしします。"
            )
        
        os.remove("temp_audio.wav")
        converted_text = replace_text(response.text, replacements)
        print(converted_text)
        return converted_text
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return ""

def save_transcription(transcription):
    try:
        with open(TRANSCRIPTION_FILE, "r+", encoding='utf-8') as file:
            data = json.load(file)
            data.append(transcription)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False)
            file.truncate()
        logger.info(f"Saved transcription: {transcription}")
    except FileNotFoundError:
        with open(TRANSCRIPTION_FILE, "w", encoding='utf-8') as file:
            json.dump([transcription], file, ensure_ascii=False)
        logger.info(f"Created new transcription file with: {transcription}")

def continuous_recording():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        logger.info("Listening...")
        while not global_state.stop_flag.is_set():
            try:
                audio = r.listen(source, phrase_time_limit=5)
                global_state.audio_queue.put(audio)
                logger.debug("Audio captured and added to queue")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in continuous recording: {str(e)}")
        logger.info("Recording stopped in continuous_recording function")

def process_audio():
    while not global_state.stop_flag.is_set() or not global_state.audio_queue.empty():
        try:
            audio = global_state.audio_queue.get(timeout=1)
            audio_data = audio.get_wav_data()
            transcription = transcribe_audio(audio_data)
            if transcription:
                save_transcription(transcription)
                global_state.transcription_queue.put(transcription)
                global_state.add_transcription(transcription)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
    logger.info("Audio processing stopped in process_audio function")

def start_background_recording():
    global_state.stop_flag.clear()
    global_state.set_recording(True)
    threading.Thread(target=continuous_recording, daemon=True).start()
    threading.Thread(target=process_audio, daemon=True).start()

def stop_background_recording():
    global_state.stop_flag.set()
    global_state.set_recording(False)
    time.sleep(2)

def clear_results():
    try:
        with open(TRANSCRIPTION_FILE, "w", encoding='utf-8') as file:
            json.dump([], file)
        global_state.clear_transcriptions()
        global_state.clear_summary()
        st.session_state.transcriptions = []
        st.session_state.summary = ""
        st.success("書き起こし結果と要約を消去しました。")
    except Exception as e:
        logger.error(f"Error clearing results: {str(e)}")
        st.error(f"結果の消去中にエラーが発生しました: {str(e)}")

def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは優秀な文章要約者です。与えられたテキストを構造的にわかりやすく要約してください。"},
                {"role": "user", "content": f"以下のテキストを要約してください:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return "要約中にエラーが発生しました。"

def main():
    st.title("リアルタイム継続音声書き起こし・要約アプリ")
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("録音開始"):
            if not global_state.get_recording():
                start_background_recording()
                st.success("バックグラウンドで録音を立ち上げました。文字起こしまで5-10秒程度お待ちください")

    with col2:
        if st.button("録音停止"):
            if global_state.get_recording():
                stop_background_recording()
                st.info("録音を停止しました。")

    with col3:
        if st.button("要約"):
            full_text = " ".join(global_state.get_transcriptions())
            summary = summarize_text(full_text)
            global_state.set_summary(summary)
            st.session_state.summary = summary
            st.success("要約が完了しました。")

    with col4:
        if st.button("結果を消去"):
            clear_results()

    st.write(f"録音状態: {'録音中' if global_state.get_recording() else '停止中'}")
    
    st.subheader("リアルタイム書き起こし結果:")
    transcription_container = st.empty()
    
    st.subheader("要約結果:")
    summary_container = st.empty()

    while True:
        new_transcriptions = global_state.get_transcriptions()
        st.session_state.transcriptions = new_transcriptions
        transcription_container.write("\n".join(st.session_state.transcriptions))
        
        summary = global_state.get_summary()
        st.session_state.summary = summary
        summary_container.write(st.session_state.summary)
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()