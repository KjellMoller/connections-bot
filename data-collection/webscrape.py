from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import time
import csv

# Make sure chromedriver is downloaded, and wherever it is stored to your export path
driver = webdriver.Chrome()
archive_puzzle_number = 1
csv_file_path = "connections_data.csv"

# Function to write data into csv file
def write_to_csv(id, data, file_path, write_headers=False):
    with open(file_path, "a", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        if write_headers:
            csvwriter.writerow(["Id","Description", "Words", "Colour"])
        for answer in data:
            csvwriter.writerow([id, answer["description"], ", ".join(answer["words"]), answer["color"]])

write_to_csv(0, [], csv_file_path, write_headers=True)

# Iterate through the entire connections archive and add to the csv file
while True:
    try:
        driver.get(f"https://connections.swellgarfo.com/nyt/{archive_puzzle_number}")
        answer_location = driver.find_element(By.ID, "__NEXT_DATA__")
        json_content = json.loads(answer_location.get_attribute("innerText"))
        if(json_content.get("props", {}).get("pageProps", {}).get("statusCode", 200) == 500):
            break
        pageProps = json_content.get("props", {}).get("pageProps", {})
        puzzle_id = json_content.get("props", {}).get("pageProps", {}).get("id", -1)
        data = json_content.get("props", {}).get("pageProps", {}).get("answers", {})
        write_to_csv(puzzle_id, data, csv_file_path)
    except Exception as e:
        print(e)
        break
    archive_puzzle_number+=1

driver.quit()