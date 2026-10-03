import os
import re
import json
import time
import random

import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================================
# 設定
# ============================================================

# ★比較するモデルに合わせて変更
# ファインチューニングしたモデルのベースモデルと同じものを使用する
BASE_MODEL_ID = "line-corporation/japanese-large-lm-1.7b-instruction-sft"

# ★LoRAアダプタ
ADAPTER_PATH = "./../final-domain-model_20260820"

# "base"       : ベースモデル
# "finetuned"  : ファインチューニング後
MODEL_TYPE = "base"

# 入力CSV
INPUT_CSV = "./results_find_1.csv"

# 出力ファイル
if MODEL_TYPE == "base":
    OUTPUT_CSV = "./inference_questions_base_20260919.csv"
    OUTPUT_JSON = "./inference_questions_base_20260919.json"

elif MODEL_TYPE == "finetuned":
    OUTPUT_CSV = "./inference_questions_finetuned_20260919.csv"
    OUTPUT_JSON = "./inference_questions_finetuned_20260919.json"

else:
    raise ValueError("MODEL_TYPE は 'base' または 'finetuned' にしてください。")


# ============================================================
# 生成設定
# ============================================================

MAX_NEW_TOKENS = 512

DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2

# ★再現性確保
SEED = 42


# ============================================================
# 乱数固定
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# モデル読み込み
# ============================================================

print("モデル初期化中...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4bit量子化設定
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ============================================================
# ベースモデル読み込み
# ============================================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)


# ============================================================
# ファインチューニングモデル読み込み
# ============================================================

if MODEL_TYPE == "finetuned":

    print("LoRAアダプタを読み込んでいます...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    print("ファインチューニング後モデルを使用します。")

else:

    model = base_model

    print("ベースモデルを使用します。")


model.eval()


# ============================================================
# プロンプト作成
# ============================================================

def create_prompt(term, explanation):

    user_input = f"""以下の用語と説明をもとに、基本情報技術者試験形式の4択問題を1問作成してください。

用語：
{term}

説明：
{explanation}

【作成条件】
・問題は1問のみ作成する
・選択肢はA、B、C、Dの4つとする
・正解は1つだけとする
・入力された説明の内容に基づいて問題を作成する
・説明にない情報を根拠として問題を作成しない
・空欄補充形式にはしない
・日本語のみを使用する
・問題、選択肢、正解、解説の順番で出力する
・解説では、正解である理由を簡潔に説明する
・解説まで出力したら生成を終了する
・修正版や別の問題を作成しない
・JSON形式を追加で出力しない
・Markdown形式を使用しない
・Human: や Assistant: を出力しない
・追加の説明や指示文を出力しない

【出力形式】
問題：
A：
B：
C：
D：
正解：
解説：

解説の出力が終了したら、そこで生成を終了してください。"""

    # ★学習時と同じ形式
    prompt = f"""### 指示:
{user_input}

### 回答:"""

    return prompt


# ============================================================
# 中国語検出
# ============================================================

def detect_chinese_text(text):
    """
    中国語の混入を検出する。

    日本語にも漢字が含まれるため、
    単純な「漢字がある = 中国語」とは判定しない。

    中国語で使われやすい文字・表現を中心に検出する。
    """

    chinese_chars = re.findall(
        r"[\u4e00-\u9fff]",
        text
    )

    # 日本語で頻繁に使われる文字以外の漢字を確認
    japanese_common = set(
        "一丁七万丈三上下不与且世両並中丸丹主久乗九乱乳乾亀了予事二互五井亜亡交享京亭人仁今介仏仕他付仙代令以仮仲件任企伊伍伏伐休会伝伯伴伸伺似但位低住佐体何余作佳併使例侍供依価侮侯侵便係促俊俗保信修俳倉個倍候借倣値倫偏停健側偵偶傍備催債傷僕僚優元兄充兆先光克免児党入全八公六共兵具典内円冊再冒冗写冬冷凍凝凡処出刀分切刊刑列初判別利到制刷券則前副割力助労効勇勉動務勝化北区医十千午半南単博印危即却原厳去参又友反収取受口古句可台史右号司各合同名向君否含吸吹周味呼命和品員哲商問啓善喪営器四回因団困囲図固国圧在地坂均型垂埋城域基堅報場塁境増墨士壮声壊壱売変外多夜夢大天太夫央失奇奈奉契奥女奴好妥妨妹妻姉始姓委威娃娘嫌子字存孫学宅宇守安完宗官定宝実客宣室宮害家容宿寂寄密富寒察審寺対寿封専射小少尚就尺尼局居届屋展属層山岐岩岳岸峡峰島川州巡巣工左巧差己巻市布帆希帝師席帯帰帳常幹平年幸幾広序底店府度座庫庭康延建弁式引弟弧弱強当形彩影役往待律後従得御復微徳徴心必応快念怒思急性怪恒恩息恐恥恵悪悲情惑惜意愛感慮慰憂憤憾懐成我戒戦戸戻所手才打払扱扶承技抄把抗折抜択披抱抵押招拝拠拡拾持指挑挙捨据掃授掌排掘掛採探接推措掲描提換握援損搬携摘摩撃攻放政故敏救敗教敢散敬数整文斗料斜断新方施旅族日旧早旬昆昇明易昔星映春昼時晩普景晴暇暖暗暮暴曜曲書最月有服望朝期木未末本札材村来東林果枝枠枚染核根格案梅械極楽概構様標模権横樹機欠次欲歌止正武歳死残段殺母毎比毛氏民気水永氷汎汗江池決汽沈沖沢河油治沿況泉法波注泳洋洗活派流浅浪浮浴海消液深混添清渇済源準溝漢演濃濁火灯灰災炉点為無焦然焼照熟熱燃父片版物特犬犯状狂独狼現球理環生産用田由甲申男町画界畑番異疑病痛癒発登白百的皆皇皮皿益盛目直相省看県真眠眼着矢知短石砂研破硬確示礼社祖祝神票祭禁私秋科秒種積穂穴空窓立章童端競竹笑第筆等筋算管節範築簡米粋精系約紅紛素経結給統絵絶継緑線編練縦縮績繁織罪置署羊美群義羽翌習考者耳聴職肉肝肥肺胃胆背胸能脂脳腸腹臓臣自至興舌船良色花芸若苦英茶草荷華落葉著蒸蔵薄薦虎虫蚊蚕行衛衣表被裁装裏補製複襟西要覆見規視覚解言計訊記討訓託訟訪設許訳証評詞詠試詩話語誤説読課調談論諸謀講謝識警議護谷豆豊貧責費貿賃資賛質赤走起越足路身車軍転軽載輸近返述迷追退送逃逆透通速造連進遊運過道達違遠選那部配酒酸里重野量金針鉄鉛銀長門閉開間関防阻降限陸険陽隠集難雨雪電需震青非面革音響頁頂順領頭題額顔風飛食飲館首香馬駅駆騎高魚鳥鳴鶏鹿麦黄黒黙"
    )

    non_japanese_kanji = [
        ch for ch in chinese_chars
        if ch not in japanese_common
    ]

    # 中国語でよく現れる語・表現
    chinese_patterns = [
        r"因此",
        r"所以",
        r"正确",
        r"错误",
        r"选项",
        r"答案",
        r"问题",
        r"解释",
        r"以下",
        r"其中",
        r"一个",
        r"可以",
        r"通过",
        r"根据",
        r"属于",
        r"表示",
        r"称为",
        r"这种",
        r"该",
        r"并且",
        r"或者",
        r"以及",
        r"分别",
        r"例如",
        r"其中",
        r"贝努利",
        r"二项分布",
    ]

    pattern_found = any(
        re.search(pattern, text)
        for pattern in chinese_patterns
    )

    # 中国語特有の句読点
    chinese_punctuation = bool(
        re.search(r"[，。！？；：]", text)
    )

    # 中国語候補
    has_non_japanese_kanji = len(non_japanese_kanji) >= 2

    return (
        pattern_found
        or has_non_japanese_kanji
        or chinese_punctuation
    )


# ============================================================
# Human / Assistant 検出
# ============================================================

def detect_human_assistant(text):

    return bool(
        re.search(
            r"(^|\n)\s*(Human|Assistant)\s*:",
            text,
            flags=re.IGNORECASE
        )
    )


# ============================================================
# 修正版・追加問題などの検出
# ============================================================

def detect_revision(text):

    patterns = [
        r"修正版",
        r"最終版",
        r"改善版",
        r"追加問題",
        r"もう一問",
        r"別の問題",
        r"別問題",
        r"もう一つ",
        r"再度",
        r"改めて",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# ============================================================
# Markdown / JSON検出
# ============================================================

def detect_markdown_json(text):

    markdown = bool(
        re.search(
            r"```|^#+\s",
            text,
            flags=re.MULTILINE
        )
    )

    json_like = bool(
        re.search(
            r'^\s*\{.*\}\s*$',
            text,
            flags=re.DOTALL
        )
    )

    return markdown or json_like


# ============================================================
# 英語などの混入検出
# ============================================================

def detect_foreign_text(text):

    # Human / Assistant は別項目で扱う
    temp = re.sub(
        r"Human:|Assistant:",
        "",
        text,
        flags=re.IGNORECASE
    )

    # ある程度長い英単語列
    english_words = re.findall(
        r"\b[A-Za-z]{4,}\b",
        temp
    )

    # 一般的な日本語内の英単語を除外
    common_words = {
        "CPU",
        "GPU",
        "HTTP",
        "HTTPS",
        "HTML",
        "JSON",
        "CSV",
        "LLM",
        "API",
        "SQL",
        "Linux",
        "Windows",
        "Python",
        "Java",
        "JavaScript",
        "QuickSort",
    }

    foreign_words = [
        word for word in english_words
        if word not in common_words
    ]

    return len(foreign_words) >= 3


# ============================================================
# 出力のクリーニング
# ============================================================

def clean_response(response):

    # ★重要
    # raw_outputそのものは変更しない。
    # この関数は「解析用」のcleaned_outputを作るだけ。

    text = response.strip()

    # 最初の「問題：」以前を削除
    match = re.search(
        r"問題\s*[:：]",
        text
    )

    if match:
        text = text[match.start():]

    # Human / Assistant以降を削除
    text = re.split(
        r"\n\s*(Human|Assistant)\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # ### 指示 / ### 回答以降
    text = re.split(
        r"\n\s*###\s*(指示|回答)\s*:",
        text,
        maxsplit=1
    )[0]

    # 修正版など以降
    text = re.split(
        r"\n\s*(修正版|最終版|改善版|追加問題|もう一問|別の問題|別問題)",
        text,
        maxsplit=1
    )[0]

    # Markdownコードブロック
    text = re.split(
        r"\n\s*```",
        text,
        maxsplit=1
    )[0]

    return text.strip()


# ============================================================
# 問題解析
# ============================================================

def parse_question(text):

    result = {
        "question": "",
        "A": "",
        "B": "",
        "C": "",
        "D": "",
        "answer": "",
        "explanation": "",
    }

    # 問題
    match = re.search(
        r"問題\s*[:：]\s*(.*?)(?=\n\s*A\s*[:：])",
        text,
        flags=re.DOTALL
    )

    if match:
        result["question"] = match.group(1).strip()

    # A
    match = re.search(
        r"\n\s*A\s*[:：]\s*(.*?)(?=\n\s*B\s*[:：])",
        text,
        flags=re.DOTALL
    )

    if match:
        result["A"] = match.group(1).strip()

    # B
    match = re.search(
        r"\n\s*B\s*[:：]\s*(.*?)(?=\n\s*C\s*[:：])",
        text,
        flags=re.DOTALL
    )

    if match:
        result["B"] = match.group(1).strip()

    # C
    match = re.search(
        r"\n\s*C\s*[:：]\s*(.*?)(?=\n\s*D\s*[:：])",
        text,
        flags=re.DOTALL
    )

    if match:
        result["C"] = match.group(1).strip()

    # D
    match = re.search(
        r"\n\s*D\s*[:：]\s*(.*?)(?=\n\s*正解\s*[:：])",
        text,
        flags=re.DOTALL
    )

    if match:
        result["D"] = match.group(1).strip()

    # 正解
    match = re.search(
        r"\n\s*正解\s*[:：]\s*([ABCDＡＢＣＤ])",
        text,
        flags=re.IGNORECASE
    )

    if match:
        answer = match.group(1).upper()

        # 全角英字を半角に変換
        answer = answer.translate(
            str.maketrans(
                "ＡＢＣＤ",
                "ABCD"
            )
        )

        result["answer"] = answer

    # 解説
    match = re.search(
        r"\n\s*解説\s*[:：]\s*(.*)",
        text,
        flags=re.DOTALL
    )

    if match:
        result["explanation"] = match.group(1).strip()

    return result


# ============================================================
# 完全性判定
# ============================================================

def check_complete(parsed):

    required_fields = [
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "explanation"
    ]

    return all(
        parsed[field].strip()
        for field in required_fields
    )


# ============================================================
# 正解判定
# ============================================================

def check_answer_valid(parsed):

    return parsed["answer"] in ["A", "B", "C", "D"]


# ============================================================
# Parse Status
# ============================================================

def determine_parse_status(
    parsed,
    reached_max_tokens,
    flags
):

    if not check_complete(parsed):
        return "INCOMPLETE"

    if not check_answer_valid(parsed):
        return "INVALID_ANSWER"

    if reached_max_tokens:
        return "MAX_TOKEN"

    if flags["has_human_assistant"]:
        return "EXTRA_DIALOGUE"

    if flags["has_revision"]:
        return "EXTRA_GENERATION"

    if flags["has_markdown_json"]:
        return "MARKDOWN_OR_JSON"

    if flags["has_chinese_text"]:
        return "CHINESE_CONTAMINATION"

    if flags["has_foreign_text"]:
        return "FOREIGN_TEXT"

    return "OK"


# ============================================================
# 入力CSV読み込み
# ============================================================

print()
print("入力CSVを読み込んでいます...")

df = pd.read_csv(INPUT_CSV)

print(f"入力データ数：{len(df)}")

print("カラム：")
print(df.columns.tolist())


# ============================================================
# 結果保存用
# ============================================================

results = []


# ============================================================
# 生成開始
# ============================================================

for index, row in df.iterrows():

    print()
    print("=" * 70)
    print(f"{index + 1} / {len(df)}")
    print("=" * 70)

    # --------------------------------------------------------
    # 入力データ
    # --------------------------------------------------------

    term = str(
        row.get("term", "")
    )

    # summary / explanation のどちらにも対応
    if "explanation" in df.columns:
        explanation_input = str(
            row.get("explanation", "")
        )

    elif "summary" in df.columns:
        explanation_input = str(
            row.get("summary", "")
        )

    else:
        explanation_input = ""

    print(f"用語：{term}")


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = create_prompt(
        term,
        explanation_input
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # 生成
    # --------------------------------------------------------

    start_time = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generation_time = time.time() - start_time


    # --------------------------------------------------------
    # Token数
    # --------------------------------------------------------

    input_tokens = inputs["input_ids"].shape[1]

    output_tokens = (
        outputs.shape[1] - input_tokens
    )


    # --------------------------------------------------------
    # 出力部分だけ取得
    # --------------------------------------------------------

    generated_tokens = outputs[
        0,
        input_tokens:
    ]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------------
    # MAX_TOKEN判定
    # --------------------------------------------------------

    reached_max_tokens = (
        output_tokens >= MAX_NEW_TOKENS
    )


    # --------------------------------------------------------
    # フラグ検出
    # --------------------------------------------------------

    flags = {

        "has_human_assistant":
            detect_human_assistant(raw_output),

        "has_revision":
            detect_revision(raw_output),

        "has_markdown_json":
            detect_markdown_json(raw_output),

        "has_chinese_text":
            detect_chinese_text(raw_output),

        "has_foreign_text":
            detect_foreign_text(raw_output),
    }


    # --------------------------------------------------------
    # 解析用cleaned_output
    # --------------------------------------------------------

    cleaned_output = clean_response(
        raw_output
    )


    # --------------------------------------------------------
    # 問題解析
    # --------------------------------------------------------

    parsed = parse_question(
        cleaned_output
    )


    # --------------------------------------------------------
    # 完全性
    # --------------------------------------------------------

    is_complete = check_complete(
        parsed
    )

    answer_valid = check_answer_valid(
        parsed
    )


    # --------------------------------------------------------
    # Parse Status
    # --------------------------------------------------------

    parse_status = determine_parse_status(
        parsed,
        reached_max_tokens,
        flags
    )


    # --------------------------------------------------------
    # 結果保存
    # --------------------------------------------------------

    result = {

        # -----------------------------
        # 入力
        # -----------------------------

        "index":
            index,

        "term":
            term,

        "explanation_input":
            explanation_input,


        # -----------------------------
        # 出力
        # -----------------------------

        "raw_output":
            raw_output,

        "cleaned_output":
            cleaned_output,


        # -----------------------------
        # 問題
        # -----------------------------

        "question":
            parsed["question"],

        "A":
            parsed["A"],

        "B":
            parsed["B"],

        "C":
            parsed["C"],

        "D":
            parsed["D"],

        "answer":
            parsed["answer"],

        "explanation":
            parsed["explanation"],


        # -----------------------------
        # 評価
        # -----------------------------

        "parse_status":
            parse_status,

        "is_complete":
            is_complete,

        "answer_valid":
            answer_valid,

        "reached_max_tokens":
            reached_max_tokens,

        "has_human_assistant":
            flags["has_human_assistant"],

        "has_revision":
            flags["has_revision"],

        "has_markdown_json":
            flags["has_markdown_json"],

        "has_chinese_text":
            flags["has_chinese_text"],

        "has_foreign_text":
            flags["has_foreign_text"],


        # -----------------------------
        # 生成情報
        # -----------------------------

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "generation_time_sec":
            round(
                generation_time,
                3
            ),
    }


    results.append(result)


    # ========================================================
    # 途中保存
    # ========================================================

    # CSV
    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )


    # JSON
    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------

    print()
    print("生成結果：")
    print(raw_output[:1000])

    print()
    print("----- 評価情報 -----")

    print(
        f"parse_status          : {parse_status}"
    )

    print(
        f"is_complete           : {is_complete}"
    )

    print(
        f"answer_valid          : {answer_valid}"
    )

    print(
        f"reached_max_tokens    : {reached_max_tokens}"
    )

    print(
        f"has_human_assistant   : "
        f"{flags['has_human_assistant']}"
    )

    print(
        f"has_revision          : "
        f"{flags['has_revision']}"
    )

    print(
        f"has_markdown_json     : "
        f"{flags['has_markdown_json']}"
    )

    print(
        f"has_chinese_text     : "
        f"{flags['has_chinese_text']}"
    )

    print(
        f"has_foreign_text     : "
        f"{flags['has_foreign_text']}"
    )

    print(
        f"input_tokens          : {input_tokens}"
    )

    print(
        f"output_tokens         : {output_tokens}"
    )

    print(
        f"generation_time_sec   : "
        f"{generation_time:.3f}"
    )


# ============================================================
# 最終保存
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 全体統計
# ============================================================

print()
print("=" * 70)
print("生成終了")
print("=" * 70)

print(
    f"CSV ：{OUTPUT_CSV}"
)

print(
    f"JSON：{OUTPUT_JSON}"
)

print()
print("===== 全体統計 =====")

total = len(results)

if total > 0:

    complete_count = sum(
        r["is_complete"]
        for r in results
    )

    chinese_count = sum(
        r["has_chinese_text"]
        for r in results
    )

    dialogue_count = sum(
        r["has_human_assistant"]
        for r in results
    )

    revision_count = sum(
        r["has_revision"]
        for r in results
    )

    max_token_count = sum(
        r["reached_max_tokens"]
        for r in results
    )

    foreign_count = sum(
        r["has_foreign_text"]
        for r in results
    )

    print(
        f"総生成数              : {total}"
    )

    print(
        f"完全生成数            : {complete_count}"
    )

    print(
        f"完全生成率            : "
        f"{complete_count / total:.3f}"
    )

    print(
        f"中国語混入数          : {chinese_count}"
    )

    print(
        f"中国語混入率          : "
        f"{chinese_count / total:.3f}"
    )

    print(
        f"Human/Assistant混入数 : {dialogue_count}"
    )

    print(
        f"修正版等混入数        : {revision_count}"
    )

    print(
        f"MAX_TOKEN到達数       : {max_token_count}"
    )

    print(
        f"MAX_TOKEN到達率       : "
        f"{max_token_count / total:.3f}"
    )

    print(
        f"外国語混入数          : {foreign_count}"
    )

    print(
        f"外国語混入率          : "
        f"{foreign_count / total:.3f}"
    )

print()
print("CSVとJSONへの保存が完了しました。")