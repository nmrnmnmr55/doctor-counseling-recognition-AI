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
import re
import atexit

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('app_debug.log', encoding='utf-8', mode='w')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# 終了処理用のフラグ
should_exit = threading.Event()

def cleanup():
    """アプリケーション終了時のクリーンアップ処理"""
    try:
        logger.info("Cleaning up resources...")
        should_exit.set()
        if global_state.get_recording():
            stop_background_recording()
        
        # 一時ファイルの削除
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        
        # キューのクリア
        while not global_state.audio_queue.empty():
            try:
                global_state.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        while not global_state.transcription_queue.empty():
            try:
                global_state.transcription_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# 終了時の処理を登録
atexit.register(cleanup)

# フィルタリングすべきフレーズのリスト
FILTERED_PHRASES = [
    "医師の会話を正確に文字起こしします",
    "患者と医師の会話を正確に文字起こしします",
    "これは医療相談の音声です",
    "本日はご覧いただきありがとうございます",
    "ありがとうございました",
    "よろしくお願いいたします",
    "ご来院ありがとうございます",
]

# 定型句を検出する正規表現パターン
COMMON_PATTERNS = [
    r'(本日は|今日は).*(ありがとうございます|ありがとうございました)',
    r'(よろしく|宜しく).*(お願いします|お願いいたします)',
    r'(ご来院|ご相談).*(ありがとうございます|ありがとうございました)',
]

def replace_text(text, replacements):
    for old, new in replacements.items():
        if new is None:
            new = ""
        text = text.replace(old, str(new))
    return text

# Read replacements from an Excel file
try:
    replacements_df = pd.read_excel('replacements.xlsx')
    replacements = dict(zip(replacements_df['old'], replacements_df['new']))
except Exception as e:
    logger.warning(f"Could not load replacements.xlsx: {e}")
    replacements = {}

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TRANSCRIPTION_FILE = "transcriptions.json"

def is_valid_transcription(text, filtered_phrases):
    if not text or len(text.strip()) < 6:
        logger.warning("Transcription too short")
        return False

    text_lower = text.lower()
    
    for phrase in filtered_phrases:
        if phrase.lower() in text_lower:
            logger.warning(f"Transcription contains filtered phrase: {phrase}")
            return False
    
    for pattern in COMMON_PATTERNS:
        if re.search(pattern, text):
            logger.warning(f"Transcription contains common pattern matching: {pattern}")
            return False
    
    common_endings = ["ありがとうございます", "ありがとうございました", "お願いいたします"]
    for ending in common_endings:
        if text.strip().endswith(ending):
            logger.warning(f"Transcription ends with common phrase: {ending}")
            return False
    
    return True

class ThreadSafeState:
    def __init__(self):
        self._lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.recording = False
        self.transcriptions = []
        self.summary = ""
        self.noise_threshold = None
        self.last_transcription_time = 0
        self.min_transcription_interval = 1.0

    def set_recording(self, value):
        with self._lock:
            self.recording = value

    def get_recording(self):
        with self._lock:
            return self.recording

    def can_add_transcription(self):
        current_time = time.time()
        with self._lock:
            if current_time - self.last_transcription_time >= self.min_transcription_interval:
                self.last_transcription_time = current_time
                return True
            return False

    def add_transcription(self, transcription):
        with self._lock:
            if transcription and is_valid_transcription(transcription, FILTERED_PHRASES):
                if not self.transcriptions or transcription != self.transcriptions[-1]:
                    self.transcriptions.append(transcription)
                    logger.info(f"Added valid transcription: {transcription}")
                else:
                    logger.warning("Rejected duplicate transcription")
            else:
                logger.warning(f"Rejected invalid transcription: {transcription}")

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

    def set_noise_threshold(self, value):
        with self._lock:
            self.noise_threshold = value

    def get_noise_threshold(self):
        with self._lock:
            return self.noise_threshold

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
                language="ja",
                prompt="医療、診察、カウンセリング、肌、治療",
                temperature=0.2
            )
        
        os.remove("temp_audio.wav")
        
        transcribed_text = response.text.strip()
        if not transcribed_text:
            return ""
            
        converted_text = replace_text(transcribed_text, replacements)
        
        logger.debug(f"Original transcription: {transcribed_text}")
        logger.debug(f"Converted text: {converted_text}")
        
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
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.truncate()
        logger.info(f"Saved transcription: {transcription}")
    except FileNotFoundError:
        with open(TRANSCRIPTION_FILE, "w", encoding='utf-8') as file:
            json.dump([transcription], file, ensure_ascii=False, indent=2)
        logger.info(f"Created new transcription file with: {transcription}")

def continuous_recording():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=3)
        r.dynamic_energy_threshold = True
        r.dynamic_energy_adjustment_ratio = 1.2
        r.energy_threshold = r.energy_threshold * 1.2
        global_state.set_noise_threshold(r.energy_threshold)
        logger.info(f"Noise threshold set to: {r.energy_threshold}")
        
        while not global_state.stop_flag.is_set() and not should_exit.is_set():
            try:
                logger.debug("Listening for speech...")
                audio = r.listen(source, 
                               phrase_time_limit=5,
                               timeout=2)
                
                if audio.get_raw_data() and global_state.can_add_transcription():
                    global_state.audio_queue.put(audio)
                    logger.debug("Audio captured and added to queue")
                
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in continuous recording: {str(e)}")
        logger.info("Recording stopped in continuous_recording function")

def process_audio():
    while not global_state.stop_flag.is_set() and not should_exit.is_set() or not global_state.audio_queue.empty():
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたは医療相談の専門家です。患者と医師の会話を重要なポイントを逃さずに要約してください。"},
                {"role": "user", "content": f"以下の医療相談の内容を要約してください:\n\n{text}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return "要約中にエラーが発生しました。"

def stop_application():
    """アプリケーションを安全に停止する"""
    try:
        cleanup()
        st.success("アプリケーションを終了します。ブラウザを閉じてください。")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    except Exception as e:
        logger.error(f"Error stopping application: {e}")
        st.error("アプリケーションの終了中にエラーが発生しました。")

def main():
    st.title("医療相談音声書き起こし・要約アプリ")
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.transcriptions = []
        st.session_state.summary = ""
        st.session_state.exit_requested = False
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("録音開始"):
            if not global_state.get_recording():
                start_background_recording()
                st.success("録音を開始しました。5-10秒後に文字起こしが起動します。")

    with col2:
        if st.button("録音停止"):
            if global_state.get_recording():
                stop_background_recording()
                st.info("録音を停止しました。")

    with col3:
        if st.button("要約"):
            full_text = " ".join(global_state.get_transcriptions())
            if full_text.strip():
                summary = summarize_text(full_text)
                global_state.set_summary(summary)
                st.session_state.summary = summary
                st.success("要約が完了しました。")
            else:
                st.warning("要約するテキストがありません。")

    with col4:
        if st.button("結果を消去"):
            clear_results()
    
    with col5:
        if st.button("終了", type="primary"):
            st.session_state.exit_requested = True
            stop_application()
            st.stop()

    if st.session_state.exit_requested:
        st.warning("アプリケーションは終了処理中です。ブラウザを閉じてください。")
        st.stop()

    st.write(f"録音状態: {'録音中' if global_state.get_recording() else '停止中'}")
    
    st.subheader("リアルタイム書き起こし結果:")
    transcription_container = st.empty()
    
    st.subheader("要約結果:")
    summary_container = st.empty()

    while not should_exit.is_set():
        try:
            new_transcriptions = global_state.get_transcriptions()
            st.session_state.transcriptions = new_transcriptions
            transcription_container.write("\n".join(st.session_state.transcriptions))
            
            summary = global_state.get_summary()
            st.session_state.summary = summary
            summary_container.write(st.session_state.summary)
            
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            break

    if should_exit.is_set():
        cleanup()
        st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup()