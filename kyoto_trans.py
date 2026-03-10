import pandas as pd
import ollama

# 1. データの読み込み（標準語10件）
df = pd.read_csv('standard_JP10.csv')
df_10 = df.head(10).copy()

def translate_with_ollama(text):
    if pd.isna(text) or text == "": return text
    
    # Ollamaへの指示（プロンプト）
    # 「標準語を、文法からしっかりした自然な京言葉に直して。出力は翻訳結果のみ。」
    prompt = f"以下の標準語の命令文を、自然な京言葉（京都弁）に翻訳して。余計な解説は不要。翻訳結果だけを出力して。\n\n標準語：{text}"
    
    try:
        
        response = ollama.generate(model='gemma3:4b', prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        print(f"エラー:{e}")
        return text

print("Ollama起動！10件、京言葉に変換")

# 2. 実行
df_10['prompt'] = df_10['prompt'].apply(translate_with_ollama)

# 3. 保存（後ろの列も全部残る）
df_10.to_csv('kyoto_ollama_10.csv', index=False, encoding='utf-8-sig')

print("完了！『kyoto_ollama_10.csv』を確認")
#print(df_10[['prompt']].head())