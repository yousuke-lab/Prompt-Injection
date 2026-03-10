import pandas as pd
from deep_translator import GoogleTranslator

# 1. データの読み込み
df = pd.read_csv('dataset.csv')
df_10 = df.head(10).copy()

def fix_to_imperative(text):
    try:
        if pd.isna(text): return ""
        
        # 翻訳実行
        translated = GoogleTranslator(source='en', target='ja').translate(str(text))
        
        # 自分の意思（〜します、〜しますか？）を、
        # システムへの指示（〜せよ、〜しろ）に強制変換するマッピング
        rules = {
            "を作成する": "を作成せよ",
            "を作成します": "を作成せよ",
            "を説明する": "を説明せよ",
            "を説明します": "を説明せよ",
            "を要約する": "を要約せよ",
            "を要約します": "を要約せよ",
            "を比較する": "を比較せよ",
            "を比較します": "を比較せよ",
            "教えます": "教えろ",
            "教えていただけますか": "教えろ",
            "アドバイスをいただけますか": "アドバイスせよ",
            "まとめます": "まとめよ",
            "の利点は何ですか": "の利点を述べよ",
            "について教えてください": "について記述せよ"
        }
        
        for old, new in rules.items():
            if old in translated:
                translated = translated.replace(old, new)
        
        # 文末が「。」や「？」で終わる不自然さを消して、命令文として完結させる
        translated = translated.rstrip("。").rstrip("？")
        
        return translated
    except:
        return text

print("10件分、指示文の文法を修正中...")

# 'prompt' 列を正しい命令形の日本語に置き換え
df_10['prompt'] = df_10['prompt'].apply(fix_to_imperative)

# 保存（後ろの列構成はそのまま）
df_10.to_csv('standard_JP10.csv', index=False, encoding='utf-8-sig')

print("完成！")
# 確認用にいくつか表示
#print(df_10[['prompt']].head(10))