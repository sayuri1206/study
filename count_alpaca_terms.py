#reviewed_json_to_alpaca.py→このファイル
import json
import csv
from pathlib import Path
from collections import Counter

# ==============================
# 設定
# ==============================

# Alpaca形式のJSONファイル
ALPACA_FILE = Path("./alpaca_train.json")

# 比較元のCSVファイル
SOURCE_CSV = Path("./data_list.csv")

# CSVの2列目を用語として使用（0始まりで1）
TERM_COLUMN = 1

# 集計結果の保存先
OUTPUT_CSV = Path("./term_adoption_counts.csv")


# ==============================
# 用語の表記を整える
# ==============================

def normalize_term(term):
    """前後の空白・改行を除去して比較する。"""
    return str(term).strip()


# ==============================
# メイン処理
# ==============================

def main():

    # ------------------------------
    # Alpaca JSONを読み込む
    # ------------------------------
    with open(ALPACA_FILE, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)

    # inputから用語を取得して問題数を数える
    adopted_counts = Counter()

    for item in alpaca_data:
        input_text = item.get("input", "")

        # 例：「用語：ベルヌーイ分布」
        if input_text.startswith("用語："):
            term = normalize_term(input_text.replace("用語：", "", 1))
            if term:
                adopted_counts[term] += 1

    # ------------------------------
    # 元のCSVから用語を読み込む
    # ------------------------------
    source_terms = []

    with open(SOURCE_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        # 1行目は見出しとしてスキップ
        next(reader, None)

        for row in reader:
            if len(row) > TERM_COLUMN:
                term = normalize_term(row[TERM_COLUMN])

                if term:
                    source_terms.append(term)

    # 同じ用語がCSVに複数ある場合は1件にまとめる
    source_terms = list(dict.fromkeys(source_terms))

    # ------------------------------
    # 採用状況を分類する
    # ------------------------------
    results = []

    one_question = []
    two_questions = []
    not_adopted = []
    other_counts = []

    for term in source_terms:
        count = adopted_counts.get(term, 0)

        if count == 1:
            status = "1問採用"
            one_question.append(term)

        elif count == 2:
            status = "2問採用"
            two_questions.append(term)

        elif count == 0:
            status = "未採用"
            not_adopted.append(term)

        else:
            status = f"{count}問採用"
            other_counts.append(term)

        results.append({
            "用語": term,
            "採用問題数": count,
            "分類": status
        })

    # ------------------------------
    # 集計結果を表示
    # ------------------------------
    print("\n===== 用語の採用状況 =====")
    print(f"CSVの用語数: {len(source_terms)}")
    print(f"1問採用の用語数: {len(one_question)}")
    print(f"2問採用の用語数: {len(two_questions)}")
    print(f"未採用の用語数: {len(not_adopted)}")
    print(f"3問以上採用の用語数: {len(other_counts)}")

    print("\n===== 1問採用 =====")
    for term in one_question:
        print(term)

    print("\n===== 2問採用 =====")
    for term in two_questions:
        print(term)

    print("\n===== 未採用 =====")
    for term in not_adopted:
        print(term)

    if other_counts:
        print("\n===== 3問以上採用（要確認） =====")
        for term in other_counts:
            print(f"{term}: {adopted_counts[term]}問")

    # ------------------------------
    # 結果をCSVに保存
    # ------------------------------
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["用語", "採用問題数", "分類"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n集計結果の保存先: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
