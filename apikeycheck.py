import os
from dotenv import load_dotenv
import openai

def debug_print(message):
    print(f"DEBUG: {message}", flush=True)

debug_print("スクリプトを開始します")

load_dotenv()
debug_print(".envファイルを読み込みました")

api_key = os.getenv("OPENAI_API_KEY")
debug_print(f"APIキーの長さ: {len(api_key) if api_key else 'APIキーが設定されていません'}")

client = openai.OpenAI(api_key=api_key)

def check_api_key():
    debug_print("APIキーのチェックを開始します")
    try:
        debug_print("OpenAIのモデルをテストします")
        
        # 簡単な完了リクエストを送信
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        
        debug_print("APIキーは有効です。レスポンス:")
        debug_print(f"- {completion.choices[0].message.content}")
        return True
    except Exception as e:
        debug_print(f"エラーが発生しました: {str(e)}")
        print(f"APIキーが無効か、エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    debug_print("メイン関数を実行します")
    check_api_key()
    debug_print("スクリプトを終了します")