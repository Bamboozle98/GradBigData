import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# === Helper function to clean currency strings ===
def clean_currency(val):
    return val.replace("$", "").replace(",", "").strip()

# === Step 1: Set up headless Chrome ===
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run in background
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=chrome_options)

# === Step 2: Loop through each page ===
all_salary_data = []
num_pages = 48  # Adjust if needed

for page in range(1, num_pages + 1):
    url = f"https://databases.usatoday.com/major-league-baseball-salaries-2024/page/{page}/"
    print(f"Scraping page {page}...")
    driver.get(url)
    time.sleep(3)  # Wait for JS content to load

    # Use BeautifulSoup on the rendered page
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # print(soup.prettify()[:1000])

    # Locate the table
    table = soup.find("table", class_="table")
    if not table:
        print(f"⚠️ No table found on page {page}")
        continue

    # Parse each row of the table
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 7:
            continue  # Skip malformed rows

        all_salary_data.append({
            "Player Name": cols[0].get_text(strip=True),
            "Team": cols[1].get_text(strip=True),
            "Position": cols[2].get_text(strip=True),
            "Salary": clean_currency(cols[3].get_text(strip=True)),
            "Years": cols[4].get_text(strip=True),
            "Total Value": clean_currency(cols[5].get_text(strip=True)),
            "Average Annual": clean_currency(cols[6].get_text(strip=True)),
        })

# === Step 3: Clean up and save ===
driver.quit()

if all_salary_data:
    headers = all_salary_data[0].keys()
    with open("../Data/mlb_salaries_2024.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_salary_data)
    print("✅ Data saved to mlb_salaries_2024.csv")
else:
    print("❌ No data was scraped.")

