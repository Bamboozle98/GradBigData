from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import time


driver = webdriver.Chrome(ChromeDriverManager().install())


# Create a Service object using the installed driver path
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Now you can use the driver as intended
driver.get("https://www.example.com")
print(driver.title)
driver.quit()


# Set up your Selenium WebDriver (adjust the path to chromedriver)
driver = webdriver.Chrome(service=Service('/path/to/chromedriver'))
all_salary_data = []
num_pages = 48

for page in range(1, num_pages + 1):
    url = f"https://databases.usatoday.com/major-league-baseball-salaries-2024/page/{page}/"
    print(f"Scraping page {page}...")
    driver.get(url)

    # Wait until the expected element is present (adjust the selector)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "db-row"))
        )
    except Exception as e:
        print(f"Timeout waiting for data on page {page}: {e}")
        continue

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    salary_entries = soup.find_all("div", class_="db-row")

    for entry in salary_entries:
        name_tag = entry.find("span", class_="player-name")
        salary_tag = entry.find("span", class_="player-salary")
        name = name_tag.get_text(strip=True) if name_tag else ""
        salary = salary_tag.get_text(strip=True) if salary_tag else ""
        all_salary_data.append({
            "Player Name": name,
            "Salary": salary,
        })

    time.sleep(1)  # Polite delay

driver.quit()

# Save data to CSV
if all_salary_data:
    headers = all_salary_data[0].keys()
    with open("../Data/mlb_salaries_2024.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in all_salary_data:
            writer.writerow(row)
    print("Data saved to mlb_salaries_2024.csv")
else:
    print("No data to save.")
