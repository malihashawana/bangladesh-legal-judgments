import requests


url = "https://supremecourt.gov.bd/web/judgments.php"


response = requests.get(url)


print(response.status_code)

print(response.text[:500])