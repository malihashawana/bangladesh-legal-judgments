import requests
from bs4 import BeautifulSoup

url = "https://supremecourt.gov.bd/web/judgments.php"

html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

print("TITLE:")
print(soup.title)

print("\nALL LINKS:\n")

count = 0

for a in soup.find_all("a"):
    print("------------------------")
    print("TEXT:", repr(a.get_text(strip=True)))
    print("HREF:", a.get("href"))
    count += 1

print("\nTotal links:", count)