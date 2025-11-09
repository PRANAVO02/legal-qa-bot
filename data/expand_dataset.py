import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin

# -------------------------------
# Configurable Sources per Domain
# -------------------------------
# SOURCES = {
#     "IPC": "https://legislative.gov.in/indian-penal-code-1860",
#     "IT": "https://www.meity.gov.in/",
#     "CONSUMER": "https://consumeraffairs.nic.in/",
#     "CRPC": "https://legislative.gov.in/",
#     "FSSAI": "https://fssai.gov.in/",
#     "POSH": "https://wcd.nic.in/",
#     "RTI": "https://rti.gov.in/",
#     "DOMESTIC_VIOLENCE": "https://legislative.gov.in/",
#     "DOWRY": "https://legislative.gov.in/",
#     "LABOUR": "https://labour.gov.in/",
#     "ENVIRONMENT": "https://moef.gov.in/",
#     "IPR": "https://ipindia.gov.in/",
#     "JUVENILE": "https://wcd.nic.in/",
#     "SC_ST": "https://socialjustice.gov.in/",
#     "CHILD_LABOUR": "https://labour.gov.in/",
#     "NI_ACT": "https://legislative.gov.in/",
#     "CONTRACT": "https://legislative.gov.in/",
#     "PROPERTY": "https://legislative.gov.in/",
#     "EVIDENCE": "https://legislative.gov.in/"
# }
SOURCES = {
    "IPC": "https://www.indiacode.nic.in/handle/123456789/2263",
    "IT": "https://www.indiacode.nic.in/handle/123456789/1999",
    "CONSUMER": "https://www.indiacode.nic.in/handle/123456789/11191",
    "CRPC": "https://www.indiacode.nic.in/handle/123456789/2264",
    "LABOUR": "https://www.indiacode.nic.in/handle/123456789/2076",
    "ENVIRONMENT": "https://www.indiacode.nic.in/handle/123456789/4314",
    "RTI": "https://www.indiacode.nic.in/handle/123456789/2065",
    "POSH": "https://www.indiacode.nic.in/handle/123456789/2079"
}

# ---------------------------------
# Helper: Extract Title + Summary
# ---------------------------------
def extract_text_from_page(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("h1") or soup.find("title")
        title_text = title.get_text(strip=True) if title else "Unknown Title"

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        clean_text = re.sub(r"\s+", " ", text)

        summary = clean_text[:600] + "..." if len(clean_text) > 600 else clean_text
        return title_text, summary
    except Exception as e:
        print(f"[!] Skipped {url}: {e}")
        return None, None

# ---------------------------------
# Generate sample section links
# ---------------------------------
def generate_section_links(domain_url, count=3):
    """
    Dummy placeholder that just generates sample-like URLs.
    You can expand this logic for real pattern scraping if site supports sections.
    """
    urls = []
    for i in range(1, count + 1):
        urls.append(domain_url)
    return urls

# ---------------------------------
# Expand Dataset
# ---------------------------------
def expand_dataset(dataset_path="legal_dataset.json", max_per_domain=3):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for domain, base_url in SOURCES.items():
        print(f"\n🔍 Expanding domain: {domain}")
        links = generate_section_links(base_url, count=max_per_domain)

        for idx, link in enumerate(links):
            title, summary = extract_text_from_page(link)
            if not title or not summary:
                continue

            new_entry = {
                "id": f"{domain.lower()}-auto-{idx+1}",
                "domain": domain,
                "title": title,
                "section": f"Auto-Section {idx+1}",
                "summary": summary,
                "source": link
            }

            # Avoid duplicates
            if not any(s["id"] == new_entry["id"] for s in data["statutes"]):
                data["statutes"].append(new_entry)
                print(f"✅ Added new section: {title[:60]}...")

            time.sleep(2)  # polite delay

    # Save updated dataset
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ Dataset expansion complete! Total statutes:", len(data["statutes"]))


# ---------------------------------
# Run
# ---------------------------------
if __name__ == "__main__":
    expand_dataset()
