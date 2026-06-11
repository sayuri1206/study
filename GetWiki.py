import wikipediaapi

# ユーザーエージェントを必ず指定（自分の連絡先やアプリ名を入れるのがマナーです）
wiki = wikipediaapi.Wikipedia(
    user_agent='HondaSayuri_study/1.0(honda3100.1206@gmail.com)',
    language='ja',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

page = wiki.page("プログレッシブウェブアプリ")
summary = page.summary.replace('\n', ' ')
url = page.fullurl

if page.exists():
    # print(f"タイトル: {page.title}\n")
    print(f"要約: {summary}\n") # 要約
    # print(f"全文: {page.text}\n") # 全文
    # 全文を取得したい場合は page.text
    print(f"URL: {url}\n")
else:
    print("ページが見つかりませんでした。")
