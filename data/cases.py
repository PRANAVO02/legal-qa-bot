import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_cases(keyword, domain, pages=2):
    base_url = "https://indiankanoon.org/search/?formInput="
    results = []
    for page in range(1, pages + 1):
        url = f"{base_url}{keyword}&pagenum={page}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")

        for item in soup.select(".result_title"):
            title = item.get_text(strip=True)
            link = "https://indiankanoon.org" + item.find("a")["href"]
            snippet_tag = item.find_next("div", class_="result_snippet")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet available."

            results.append({
                "domain": domain,
                "title": title,
                "snippet": snippet,
                "url": link
            })
        time.sleep(2)  # be polite
    return results


# Example usage:
new_cases = []
topics = {
    "IPC": ["theft", "murder", "rape", "cheating"],
    "IT": ["cybercrime", "identity theft", "privacy"],
    "CONSUMER": ["refund", "defective product", "compensation"],
    "LABOUR": ["minimum wages", "employer liability"]
}

for domain, keywords in topics.items():
    for kw in keywords:
        print(f"Scraping {domain}: {kw}")
        new_cases.extend(scrape_cases(kw, domain))

with open("expanded_cases.json", "w", encoding="utf-8") as f:
    json.dump(new_cases, f, indent=2, ensure_ascii=False)

print(f"✅ Total new cases added: {len(new_cases)}")
