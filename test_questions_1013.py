import time
import json
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import unicodedata

# =====================================
# ファイル名設定
# =====================================

# 用語・要約・問題・選択肢などを入力したCSV
INPUT_CSV = "test_questions.csv"

# 判定結果の保存先
RESULT_JSON = "test_results_20261013.json"
RESULT_CSV = "test_results_20261013.csv"

# =====================================
# モデル読み込み
# =====================================

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto"
)

# =====================================
# 類似度モデル
# =====================================

sim_model = SentenceTransformer(
    "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
)

SIM_THRESHOLD = 0.6

timestart = time.time()

# =====================================
# CSV読み込み
# =====================================

df = pd.read_csv(INPUT_CSV, keep_default_na=False)

# 必須列を確認
required_columns = [
    "term", "summary", "question",
    "A", "B", "C", "D", "answer", "explanation"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"入力CSVに必要な列がありません: {missing_columns}\n"
        f"必要な列: {required_columns}"
    )

# 文字列として扱い、前後の空白を除去
for column in required_columns:
    df[column] = df[column].astype(str).str.strip()

print(f"\nテスト対象の問題数: {len(df)}")

# =====================================
# Python判定用関数
# =====================================

THRESHOLD = 15

placeholders = [
    "問題文",
    "選択肢A",
    "選択肢B",
    "選択肢C",
    "選択肢D",
    "解説文"
]


def japanese_ratio(text):
    if pd.isna(text):
        return 0

    text = str(text)

    if "<END>" in text:
        text = text.split("<END>")[0]

    if len(text.strip()) == 0:
        return 0

    japanese = 0
    target = 0

    for c in text:
        if c.isspace():
            continue

        code = ord(c)

        # ひらがな
        if 0x3040 <= code <= 0x309F:
            japanese += 1
            target += 1

        # カタカナ
        elif 0x30A0 <= code <= 0x30FF:
            japanese += 1
            target += 1

        # 漢字
        elif unicodedata.name(c, "").startswith("CJK UNIFIED"):
            japanese += 1
            target += 1

        # 英数字
        elif c.isascii() and c.isalnum():
            target += 1

    if target == 0:
        return 0

    return japanese / target


def is_alnum_only(text):
    text = str(text)

    return re.fullmatch(
        r"[A-Za-z0-9 ]+",
        text
    ) is not None


def contains_formula(text):
    text = str(text)

    pattern = (
        r"[=+\-*/^√∑±<>]"
        r"|[Pp]\("
        r"|log"
        r"|sin|cos|tan"
        r"|[xyzXYZ]\^"
        r"|[0-9]+\.[0-9]+"
    )

    return re.search(pattern, text) is not None


def contains_chinese(text):
    if pd.isna(text):
        return False

    text = str(text)

    if "<END>" in text:
        text = text.split("<END>")[0]

    has_kana = re.search(r"[ぁ-んァ-ヶ]", text)
    has_kanji = re.search(r"[\u4e00-\u9fff]", text)

    return bool(has_kanji and not has_kana)


# =====================================
# 各入力問題を判定
# =====================================

results = []

for index, row in df.iterrows():
    term = row["term"]
    summary = row["summary"]
    question = row["question"]
    choice_a = row["A"]
    choice_b = row["B"]
    choice_c = row["C"]
    choice_d = row["D"]
    answer = row["answer"]
    explanation = row["explanation"]

    print(f"\n判定中: {term} ({index + 1}/{len(df)})")

    # 各問題の結果を初期化
    reject = False
    reason = []

    python_reject = False
    python_reason = []
    llm_reject = None
    llm_reason = ""
    json_error = None
    review_generated_text = ""
    review_input_tokens = 0
    review_output_tokens = 0
    sim = None
    sim_reject = None
    final_reject = None
    final_reason = ""

    # =====================================
    # 必須項目
    # =====================================

    required_fields_for_check = [
        ("question", question),
        ("A", choice_a),
        ("B", choice_b),
        ("C", choice_c),
        ("D", choice_d),
        # 元コードと同じく answer は必須項目チェック対象外
        ("explanation", explanation)
    ]

    for name, value in required_fields_for_check:
        if pd.isna(value) or len(str(value).strip()) == 0:
            reject = True
            reason.append(f"{name}が空")

    # =====================================
    # プレースホルダ
    # =====================================

    for name, value in required_fields_for_check:
        if str(value).strip() in placeholders:
            reject = True
            reason.append(f"{name}がプレースホルダ")

    # =====================================
    # 日本語率
    # =====================================

    japanese_ratios = {}

    for name, value in required_fields_for_check:
        ratio = japanese_ratio(value)
        japanese_ratios[f"japanese_ratio_{name}"] = ratio

        if ratio < 0.6:
            reject = True
            reason.append(f"{name}の日本語率が低い")

    # =====================================
    # 中国語らしい文章
    # =====================================

    for name, value in required_fields_for_check:
        if contains_chinese(value):
            reject = True
            reason.append(f"{name}が中国語の可能性")

    # =====================================
    # 英数字のみ
    # =====================================

    for name, value in required_fields_for_check:
        if is_alnum_only(value):
            reject = True
            reason.append(f"{name}が英数字のみ")

    # =====================================
    # 数式
    # =====================================

    check_targets = {
        "question": question,
        "A": choice_a,
        "B": choice_b,
        "C": choice_c,
        "D": choice_d
    }

    for target, value in check_targets.items():
        if contains_formula(value):
            reject = True
            reason.append(f"{target}に数式")

    # =====================================
    # 選択肢文字数
    # =====================================

    lengths = {
        "A": len(str(choice_a)),
        "B": len(str(choice_b)),
        "C": len(str(choice_c)),
        "D": len(str(choice_d))
    }

    for target in ["A", "B", "C", "D"]:
        others = [
            value
            for key, value in lengths.items()
            if key != target
        ]

        average = sum(others) / 3

        if abs(lengths[target] - average) > THRESHOLD:
            reject = True
            reason.append(f"{target}の文字数が大きく異なる")

    python_reject = reject
    python_reason = list(reason)

    # =====================================
    # LLM添削・類似度判定
    # Python判定で却下された問題も、両方の判定を実行する
    # =====================================
    review = {
        "reject": False,
        "reason": []
    }

    json_error = False

    review_prompt = f"""
あなたは基本情報技術者試験の問題を添削する専門家です。

以下の問題を評価してください。

【用語】
{term}

【用語の説明】
{summary}

【問題】
{question}

【選択肢】
A {choice_a}
B {choice_b}
C {choice_c}
D {choice_d}

【正解】
{answer}

【解説】
{explanation}

以下のみ判定してください。

・問題が用語に対応しているか
・問題として自然か
・正解と解説に矛盾がないか

JSONのみ出力してください。

{{
"reject": false,
"reason": []
}}
"""

    review_result = generator(
        review_prompt,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False
    )

    review_generated_text = review_result[0]["generated_text"]

    review_input_tokens = len(
        tokenizer.encode(review_prompt)
    )

    review_output_tokens = len(
        tokenizer.encode(review_generated_text)
    )

    try:
        # 余分なコードフェンスがあってもJSON本体を探す
        json_text = review_generated_text.strip()
        json_match = re.search(r"\{.*\}", json_text, re.DOTALL)

        if json_match:
            review = json.loads(json_match.group(0))
        else:
            raise json.JSONDecodeError(
                "JSON object not found",
                json_text,
                0
            )

        if not isinstance(review.get("reject"), bool):
            raise ValueError("LLMのrejectがboolではありません")

        if not isinstance(review.get("reason", []), list):
            raise ValueError("LLMのreasonが配列ではありません")

    except (json.JSONDecodeError, ValueError, TypeError):
        json_error = True
        review = {
            "reject": True,
            "reason": ["LLMの判定結果をJSONとして解析できない"]
        }

    llm_reject = review["reject"]
    llm_reason = ", ".join(map(str, review.get("reason", [])))

    # =====================================
    # summary と question の類似度
    # =====================================

    summary_embedding = sim_model.encode(
        summary,
        convert_to_tensor=True
    )

    question_embedding = sim_model.encode(
        question,
        convert_to_tensor=True
    )

    sim = util.cos_sim(
        summary_embedding,
        question_embedding
    ).item()

    # 元コードと同じ判定: sim <= 0.6 なら却下
    sim_reject = sim <= SIM_THRESHOLD

    # =====================================
    # 最終判定
    # Python判定または類似度判定で却下なら最終却下
    # LLMのrejectは元コード同様、独立した評価結果として記録する
    # =====================================

    final_reject = bool(python_reject or sim_reject)
    final_reasons = []

    if python_reject:
        final_reasons.append(f"Python判定: {', '.join(python_reason)}")

    if llm_reject:
        final_reasons.append(f"LLM判定は却下: {llm_reason}")

    if json_error:
        final_reasons.append("LLMのJSON解析エラー")

    if sim_reject:
        final_reasons.append(
            f"類似度が閾値以下 ({sim:.4f} <= {SIM_THRESHOLD})"
        )
    elif sim is not None:
        final_reasons.append(
            f"類似度が閾値を超過 ({sim:.4f} > {SIM_THRESHOLD})"
        )

    final_reason = ", ".join(final_reasons)

    # =====================================
    # 結果を保存用リストに追加
    # =====================================

    result_row = {
        "term": term,
        "summary": summary,
        "question": question,
        "A": choice_a,
        "B": choice_b,
        "C": choice_c,
        "D": choice_d,
        "answer": answer,
        "explanation": explanation,

        "python_reject": python_reject,
        "python_reason": ", ".join(python_reason),

        "llm_reject": llm_reject,
        "llm_reason": llm_reason,
        "json_error": json_error,
        "review_generated_text": review_generated_text,
        "review_input_tokens": review_input_tokens,
        "review_output_tokens": review_output_tokens,

        "sim": sim,
        "sim_threshold": SIM_THRESHOLD,
        "sim_reject": sim_reject,

        "final_reject": final_reject,
        "final_reason": final_reason
    }

    result_row.update(japanese_ratios)
    results.append(result_row)

    print(
        f"Python却下={python_reject}, "
        f"LLM却下={llm_reject}, "
        f"類似度={sim if sim is not None else '未計算'}, "
        f"最終却下={final_reject}"
    )

# =====================================
# JSON保存
# =====================================

with open(RESULT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=4
    )

# =====================================
# CSV保存
# =====================================

result_df = pd.DataFrame(results)
result_df.to_csv(
    RESULT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# 集計表示
# =====================================

python_rejected_count = sum(
    bool(row["python_reject"])
    for row in results
)

llm_judged_count = sum(
    row["llm_reject"] is not None
    for row in results
)

llm_rejected_count = sum(
    row["llm_reject"] is True
    for row in results
)

sim_judged = [
    row["sim"]
    for row in results
    if row["sim"] is not None
]

final_rejected_count = sum(
    bool(row["final_reject"])
    for row in results
)

print("\n========== 判定結果 ==========")
print(f"入力問題数: {len(results)}")
print(f"Python判定で却下: {python_rejected_count}")
print(f"LLM判定を実行した問題数: {llm_judged_count}")
print(f"LLM判定で却下: {llm_rejected_count}")
print(f"類似度を計算した問題数: {len(sim_judged)}")

if sim_judged:
    print(f"平均類似度: {sum(sim_judged) / len(sim_judged):.4f}")
    print(f"類似度閾値: {SIM_THRESHOLD}")

print(f"最終却下数: {final_rejected_count}")
print(f"最終通過数: {len(results) - final_rejected_count}")
print(f"JSON保存先: {RESULT_JSON}")
print(f"CSV保存先: {RESULT_CSV}")

# =====================================
# 処理時間
# =====================================

elapsed = time.time() - timestart

days = int(elapsed // 86400)
hours = int((elapsed % 86400) // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60

print(
    f"処理時間: {days}日 {hours}時間 "
    f"{minutes}分 {seconds:.2f}秒"
)
