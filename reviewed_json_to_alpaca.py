#このファイル→count_alpaca_terms.py
import json
from pathlib import Path
from collections import defaultdict

# ==============================
# 設定
# ==============================

# 元のJSONファイルを保存しているフォルダ
INPUT_DIR = Path("./json_data")

# 出力するAlpaca形式のJSONファイル
OUTPUT_FILE = Path("./alpaca_train.json")

# 学習時の指示文
INSTRUCTION = (
    "基本情報技術者試験の出題傾向を参考に、"
    "与えられた用語から4択問題を1問生成してください。"
    "問題文、選択肢A～D、正解を出力してください。"
)

# ==============================
# JSONファイルの読み込み
# ==============================

def load_json_data(file_path):
    """JSONファイルから問題データを取得する。"""

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]

        return [data]

    return []


# ==============================
# Alpaca形式への変換
# ==============================

def convert_to_alpaca(item):
    """元データ1件をAlpaca形式に変換する。"""

    # 却下されたデータは学習に使用しない
    if item.get("reject") is True:
        return None

    term = item.get("term", "")
    question = item.get("question", "")

    choices = {
        "A": item.get("A", ""),
        "B": item.get("B", ""),
        "C": item.get("C", ""),
        "D": item.get("D", ""),
    }

    answer = str(item.get("answer", "")).strip().upper()

    # 「B|||選択肢の文章」のような形式にも対応
    if "|||" in answer:
        answer = answer.split("|||", 1)[0].strip()

    # 必須項目が欠けているデータは除外
    if not isinstance(term, str) or not term.strip():
        return None

    if not isinstance(question, str) or not question.strip():
        return None

    if not all(
        isinstance(value, str) and value.strip()
        for value in choices.values()
    ):
        return None

    if answer not in ["A", "B", "C", "D"]:
        return None

    # 出力する問題文を作成
    output = (
        f"問題：{question}\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n"
        f"正解：{choices[answer]}"
    )

    # Alpaca形式に変換
    alpaca_data = {
        "instruction": INSTRUCTION,
        "input": f"用語：{term}",
        "output": output
    }

    return alpaca_data


# ==============================
# メイン処理
# ==============================

def main():

    alpaca_dataset = []

    # フォルダ内のJSONファイルをすべて取得
    json_files = sorted(INPUT_DIR.rglob("*.json"))

    # 出力ファイル自身は読み込まない
    json_files = [
        path for path in json_files
        if path.resolve() != OUTPUT_FILE.resolve()
    ]

    if not json_files:
        print(f"JSONファイルが見つかりません: {INPUT_DIR}")
        return

    all_data = []
    rejected_count = 0

    # ==============================
    # JSONファイルをすべて読み込む
    # ==============================
    for file_path in json_files:
        try:
            data_list = load_json_data(file_path)

            for item in data_list:
                if isinstance(item, dict):
                    all_data.append(item)
                else:
                    rejected_count += 1

            print(f"読み込み完了: {file_path}")

        except (json.JSONDecodeError, OSError) as e:
            print(f"読み込みエラー: {file_path}: {e}")

    # ==============================
    # 用語ごとに問題をまとめる
    # ==============================
    term_groups = defaultdict(list)

    for item in all_data:
        term = item.get("term", "")

        if isinstance(term, str) and term.strip():
            term_groups[term].append(item)
        else:
            rejected_count += 1

    # # 【原因調査用】ヒューリスティック法の元データ数を確認
    # for term, items in term_groups.items():
    #     if term.strip() == "ヒューリスティック法":
    #         print(f"\n用語：{term}")
    #         print(f"元データ数：{len(items)}")

    #         for i, item in enumerate(items, start=1):
    #             print(f"\n{i}件目")
    #             print(f"reject：{item.get('reject')}")
    #             print(f"問題文あり：{bool(item.get('question'))}")
    #             print(
    #                 f"A～Dあり："
    #                 f"{all(item.get(k) for k in ['A', 'B', 'C', 'D'])}"
    #             )
    #             print(f"正解：{item.get('answer')}")
    # ==============================
    # 用語ごとの問題数を調整
    # 3問以上なら先頭から2問だけ採用
    # ==============================
    selected_data = []

    for term, items in term_groups.items():

        if len(items) >= 3:
            selected_data.extend(items[:2])
        else:
            selected_data.extend(items)

    # ==============================
    # Alpaca形式に変換
    # ==============================
    for item in selected_data:

        converted = convert_to_alpaca(item)

        if converted is None:
            rejected_count += 1
            continue

        alpaca_dataset.append(converted)

    # ==============================
    # 実際に採用された問題を用語ごとに集計
    # ==============================
    final_term_counts = defaultdict(int)

    for item in alpaca_dataset:
        term = item["input"].replace("用語：", "", 1)
        final_term_counts[term] += 1

    # 2問採用された用語数
    two_question_terms = sum(
        1 for count in final_term_counts.values()
        if count == 2
    )

    # 1問採用された用語数
    one_question_terms = sum(
        1 for count in final_term_counts.values()
        if count == 1
    )

    # 3問以上採用された用語数（確認用）
    three_or_more_terms = sum(
        1 for count in final_term_counts.values()
        if count >= 3
    )

    # ==============================
    # JSONファイルに保存
    # ==============================
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            alpaca_dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ==============================
    # 結果表示
    # ==============================
    print("\n===== 変換結果 =====")
    print(f"読み込みデータ数: {len(all_data)}")
    print(f"用語数: {len(term_groups)}")
    print(f"選択したデータ数: {len(selected_data)}")
    print(f"採用データ数: {len(alpaca_dataset)}")
    print(f"2問採用した用語数: {two_question_terms}")
    print(f"1問採用した用語数: {one_question_terms}")
    print(f"3問以上採用した用語数: {three_or_more_terms}")
    print(f"除外データ数: {rejected_count}")
    print(f"保存先: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
