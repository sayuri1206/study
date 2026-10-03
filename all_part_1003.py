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

INPUT_CSV = "results_find_1.csv"

GENERATED_JSON = "generate.json"
GENERATED_CSV = "generate.csv"

REVIEWED_JSON = "reviewed_questions_20261001.json"
REVIEWED_CSV = "reviewed_questions_20261001.csv"

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

# -------------------------------------
# data.csvから対象用語を取得
# -------------------------------------

DATA_CSV = "data_list.csv"

data_df = pd.read_csv(DATA_CSV)

# 1列目を数値として扱う
data_df.iloc[:, 0] = pd.to_numeric(
    data_df.iloc[:, 0],
    errors="coerce"
)

# 1列目が5未満の行だけ取得
target_df = data_df[
    data_df.iloc[:, 0] < 5
]

# 2列目（用語）を配列に格納
terms = (
    target_df.iloc[:, 1]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

print("\n対象用語:")
for term in terms:
    print(term)


# -------------------------------------
# INPUT_CSVを読み込み
# -------------------------------------

df = pd.read_csv(INPUT_CSV)

# INPUT_CSVのinstruction列を文字列として扱う
df["instruction"] = (
    df["instruction"]
    .astype(str)
    .str.strip()
)

# terms配列に含まれる用語だけを残す
df = df[
    df["instruction"].isin(terms)
].copy()

print(
    f"\n推論対象の用語数: "
    f"{df['instruction'].nunique()}"
)

print(
    f"推論対象の行数: "
    f"{len(df)}"
)


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

    return has_kanji and not has_kana


# =====================================
# 保存用
# =====================================

# 生成直後の全問題
generated_questions = []

# Python判定・LLM添削・類似度判定後の問題
reviews = []


# =====================================
# 各単語について10問生成
# =====================================

for _, row in df.iterrows():

    term = row["instruction"]
    summary = row["output"]

    for i in range(20):

        print(
            f"\n生成中: {term} "
            f"({i + 1}/20)"
        )


        # =====================================
        # 問題生成プロンプト
        # =====================================

        prompt = f"""
    あなたは基本情報技術者試験の問題作成者です。

    以下の用語と説明を参考に、
    基本情報技術者試験の午前科目と同レベルの
    4択問題を1問作成してください。

    【用語】
    {term}

    【説明】
    {summary}

    【作問ルール】

    ・問題文は1問のみ
    ・選択肢はA～Dの4つ
    ・正解は1つのみ
    ・誤答はIT分野として自然な内容にする
    ・誤答は初学者が迷いやすい内容にする
    ・「○○とは何か」だけでなく、
    概念理解、適用場面、特徴比較、
    セキュリティ、ネットワーク、
    ソフトウェア開発なども出題対象とする
    ・基本情報技術者試験の実際の問題形式に近づける
    ・選択肢の長さはできるだけ揃える
    ・正解が明らかにならないようにする
    ・解説以降は出力禁止
    ・1問出力したら終了すること。
    ・2問目以降は絶対に出力しないこと。
    ・出力は次の形式のみとする。
    ・解説は各選択肢の正誤を説明する内容にすること。
    ・解説は500字程度にすること。
    ・解説の最後に必ず <END> を出力し、
    ・その後は何も出力しないこと。

    【重要】

    出力は必ず日本語のみを使用すること。
    英語、中国語、韓国語、その他の外国語は出力しないこと。
    問題文、選択肢、正解、解説の全てを日本語で記述すること。

    【例1】

    問題|||WAFの説明として適切なものはどれか。

    A|||Webサイトへの攻撃を検知し遮断する
    B|||無線LANの暗号化方式である
    C|||ログを一元管理するシステムである
    D|||統合脅威管理装置である

    正解|||A

    解説|||WAFはWebアプリケーションへの攻撃を検知・遮断する仕組みである。

    <END>

    【例2】

    問題|||アジャイル開発手法の一つであるスクラムにおいて，プロダクトバックログアイテムの内容や並び順を決定する役割をもつのは誰か。

    A|||開発者
    B|||顧客
    C|||スクラムマスタ
    D|||プロダクトオーナ

    正解|||D

    解説|||プロダクトオーナは、プロダクトバックログアイテムの内容や並び順を決定する役割をもつ。

    <END>

    【出力形式】

    問題|||問題文

    A|||選択肢A
    B|||選択肢B
    C|||選択肢C
    D|||選択肢D

    正解|||A

    解説|||解説文

    <END>

    上記以外の文章は出力しないこと。
    """


        # =====================================
        # 問題生成
        # =====================================

        result = generator(
            prompt,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_text = result[0]["generated_text"]


        input_tokens = len(
            tokenizer.encode(prompt)
        )

        generated_text = generated_text.split("<END>")[0]

        generated_tokens = len(
            tokenizer.encode(generated_text)
        )


        # =====================================
        # 初期化
        # =====================================

        question = ""
        choice_a = ""
        choice_b = ""
        choice_c = ""
        choice_d = ""
        answer = ""
        explanation = ""

        current_field = None


        # =====================================
        # パース
        # =====================================

        for line in generated_text.split("\n"):

            line = line.strip()

            if line.startswith("問題|||"):

                question = line.replace(
                    "問題|||", ""
                ).strip()

            elif line.startswith("A|||"):

                choice_a = line.replace(
                    "A|||", ""
                ).strip()

            elif line.startswith("B|||"):

                choice_b = line.replace(
                    "B|||", ""
                ).strip()

            elif line.startswith("C|||"):

                choice_c = line.replace(
                    "C|||", ""
                ).strip()

            elif line.startswith("D|||"):

                choice_d = line.replace(
                    "D|||", ""
                ).strip()

            elif line.startswith("正解|||"):

                answer = line.replace(
                    "正解|||", ""
                ).strip()

            if line.startswith("解説|||"):

                current_field = "explanation"

                explanation = line.replace(
                    "解説|||", ""
                ).strip()

            elif current_field == "explanation":

                explanation += "\n" + line

        # =====================================
        # 生成直後の問題を保存用リストに追加
        # =====================================

        generated_questions.append({

            "term": term,

            "summary": summary,

            "question": question,

            "A": choice_a,
            "B": choice_b,
            "C": choice_c,
            "D": choice_d,

            "answer": answer,

            "explanation": explanation,

            "input_tokens": input_tokens,

            "output_tokens": generated_tokens,

            "generated_text": generated_text,

            # 追加：Python判定・類似度判定の結果
            "python_reject": False,
            "python_reason": "",
            "sim": None,
            "sim_reject": None

        })

        # =====================================
        # 必須項目
        # =====================================

        reject = False
        reason = []

        required_fields = [
            ("question", question),
            ("A", choice_a),
            ("B", choice_b),
            ("C", choice_c),
            ("D", choice_d),
            # ("answer", answer),
            ("explanation", explanation)
        ]


        for name, value in required_fields:

            if pd.isna(value) or len(
                str(value).strip()
            ) == 0:

                reject = True

                reason.append(
                    f"{name}が空"
                )


        # =====================================
        # プレースホルダ
        # =====================================

        for name, value in required_fields:

            if str(value).strip() in placeholders:

                reject = True

                reason.append(
                    f"{name}がプレースホルダ"
                )


        # =====================================
        # 日本語率
        # =====================================

        for name, value in required_fields:

            ratio = japanese_ratio(value)

            if ratio < 0.6:

                reject = True

                reason.append(
                    f"{name}の日本語率が低い"
                )


        # =====================================
        # 中国語らしい文章
        # =====================================

        for name, value in required_fields:

            if contains_chinese(value):

                reject = True

                reason.append(
                    f"{name}が中国語の可能性"
                )


        # =====================================
        # 英数字のみ
        # =====================================

        for name, value in required_fields:

            if is_alnum_only(value):

                reject = True

                reason.append(
                    f"{name}が英数字のみ"
                )


        # =====================================
        # 数式
        # =====================================

        for target in [
            "question",
            "A",
            "B",
            "C",
            "D"
        ]:

            if contains_formula(
                locals()[{
                    "question": "question",
                    "A": "choice_a",
                    "B": "choice_b",
                    "C": "choice_c",
                    "D": "choice_d"
                }[target]]
            ):

                reject = True

                reason.append(
                    f"{target}に数式"
                )


        # =====================================
        # 選択肢文字数
        # =====================================

        lengths = {

            "A": len(str(choice_a)),
            "B": len(str(choice_b)),
            "C": len(str(choice_c)),
            "D": len(str(choice_d))
        }


        for target in [
            "A",
            "B",
            "C",
            "D"
        ]:

            others = [
                v
                for k, v in lengths.items()
                if k != target
            ]

            average = sum(others) / 3

            if abs(
                lengths[target] - average
            ) > THRESHOLD:

                reject = True

                reason.append(
                    f"{target}の文字数が大きく異なる"
                )


        # =====================================
        # Python判定で却下された場合
        # =====================================

        if reject:

            # 追加：generate.csv / generate.json にPython判定結果を保存
            generated_questions[-1]["python_reject"] = True
            generated_questions[-1]["python_reason"] = ", ".join(reason)
            generated_questions[-1]["sim"] = None
            generated_questions[-1]["sim_reject"] = None

            print(
                f"Python却下: {term} "
                f"({i + 1}/10)"
            )

            continue


        # =====================================
        # LLM添削
        # =====================================

        review = {
            "reject": False,
            "reason": []
        }

        json_error = False
        review_generated_text = ""

        review_input_tokens = 0
        review_output_tokens = 0


        review_prompt = f"""
あなたは基本情報技術者試験の問題を添削する専門家です。

以下の問題を評価してください。

【用語】
{term}

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


        review_generated_text = (
            review_result[0]["generated_text"]
        )


        review_input_tokens = len(
            tokenizer.encode(review_prompt)
        )

        review_output_tokens = len(
            tokenizer.encode(
                review_generated_text
            )
        )


        try:

            review = json.loads(
                review_generated_text
            )

        except json.JSONDecodeError:

            json_error = True


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


        # =====================================
        # sim > 0.6 の場合のみ保存
        # =====================================

        if sim <= SIM_THRESHOLD:

            # 追加：generate.csv / generate.json に類似度判定結果を保存
            generated_questions[-1]["python_reject"] = False
            generated_questions[-1]["python_reason"] = ""
            generated_questions[-1]["sim"] = sim
            generated_questions[-1]["sim_reject"] = True

            print(
                f"類似度却下: {term} "
                f"({i + 1}/10) "
                f"sim={sim:.4f}"
            )

            continue


        # =====================================
        # 最終保存
        # =====================================
        # 追加：generate.csv / generate.json に最終的な判定結果を保存
        generated_questions[-1]["python_reject"] = False
        generated_questions[-1]["python_reason"] = ""
        generated_questions[-1]["sim"] = sim
        generated_questions[-1]["sim_reject"] = False

        reviews.append({

            "term": term,

            "summary": summary,

            "question": question,

            "A": choice_a,
            "B": choice_b,
            "C": choice_c,
            "D": choice_d,

            "answer": answer,

            "explanation": explanation,

            "reject": review["reject"],

            "reason": ", ".join(
                review["reason"]
            ),

            "json_error": json_error,

            "generated_text": generated_text,

            "input_tokens": input_tokens,

            "output_tokens": generated_tokens,

            "sim": sim

        })


        print(
            f"保存: {term} "
            f"({i + 1}/10) "
            f"sim={sim:.4f}"
        )


# =====================================
# 生成直後の全問題をJSON保存
# =====================================

with open(
    GENERATED_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        generated_questions,
        f,
        ensure_ascii=False,
        indent=4
    )


# =====================================
# 生成直後の全問題をCSV保存
# =====================================

generated_df = pd.DataFrame(
    generated_questions
)

generated_df.to_csv(
    GENERATED_CSV,
    index=False,
    encoding="utf-8-sig"
)


# =====================================
# 最終採用問題をJSON保存
# =====================================

with open(
    REVIEWED_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        reviews,
        f,
        ensure_ascii=False,
        indent=4
    )


# =====================================
# 最終採用問題をCSV保存
# =====================================

reviewed_df = pd.DataFrame(
    reviews
)

reviewed_df.to_csv(
    REVIEWED_CSV,
    index=False,
    encoding="utf-8-sig"
)


# =====================================
# 保存完了
# =====================================

print("\n保存完了")

print(f"生成データ JSON: {GENERATED_JSON}")
print(f"生成データ CSV : {GENERATED_CSV}")

print(f"採用データ JSON: {REVIEWED_JSON}")
print(f"採用データ CSV : {REVIEWED_CSV}")

print(
    f"生成した問題数: "
    f"{len(generated_questions)}問"
)

print(
    f"最終保存問題数: "
    f"{len(reviews)}問"
)


# =====================================
# 処理時間
# =====================================

timeend = time.time()

elapsed = timeend - timestart

days = int(elapsed // 86400)
hours = int((elapsed % 86400) // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60

print(
    f"処理時間: "
    f"{days}日 "
    f"{hours}時間 "
    f"{minutes}分 "
    f"{seconds:.2f}秒"
)