import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Path to ChromeDriver
driver_path = r'C:\\webdrivers\\chromedriver.exe'  # Update this path
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Selenium setup
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (remove this if you want to see the browser)

# URL of the page
url = "https://www.peerspot.com/products/sap-successfactors-reviews"
driver.get(url)
time.sleep(3)  # Wait for the page to load

# Set to keep track of processed review titles
processed_titles = set()


def extract_reviews(soup):
    page_source = driver.page_source
    # print(page_source)  # This will print the raw HTML source to inspect the structure
    soup = BeautifulSoup(page_source, 'html.parser')

    reviews_data = []
    name_data = []
    title_data = []
    pros_data = []
    cons_data = []

    # Extract user details
    user_details = soup.find_all("div", class_="user-details user-details-horizontal")
    title_divs = soup.find_all('div', class_='title text-secondary font-22 0')

    # Extract the text from each div and print it
    for title_text in title_divs:
        title_data.append({
            "title": title_text.get_text(strip=True),
        })

    title_divs = soup.find_all('div', class_='text-secondary font-18 content-styling sp-line-5')

    # Extract the text from each div and print it
    for title_text in title_divs:
        pros_data.append({
            "pros_cons": title_text.get_text(strip=True),
        })

    for user_detail in user_details:
        name = user_detail.find("a", class_="author-info block-link")
        if name:
            name = name.text.strip()
        else:
            name = "Anonymous"

        name_data.append({
            "name": name,
        })

    # Now, correctly iterate over pros_data and alternate between pros and cons
    pros = []
    cons = []

    # Iterate over the pros_data list
    for i, item in enumerate(pros_data):
        pros_cons_text = item["pros_cons"]  # Accessing the pros_cons text from each dictionary
        if i % 2 == 0:
            pros.append({
            "pros": pros_cons_text})
        else:  # If index is odd, assign to cons
            cons.append({
                "cons": pros_cons_text})
    # Create the reviews_data dictionary

    reviews_data = {
        "name": name_data,
        "title": title_data,
        "pros": pros,
        "cons": cons
    }
    result = []
    min_length = min(len(reviews_data['name']), len(reviews_data['title']), len(reviews_data['pros']),
                     len(reviews_data['cons']))
    entry = {"data": []}
    for i in range(min_length):  # Ensure equal-length data
        entry = {
            'title': reviews_data['title'][i]['title'],
            'name': reviews_data['name'][i]['name'],
            'pros': reviews_data['pros'][i]['pros'],
            'cons': reviews_data['cons'][i]['cons']
        }
        result.append(entry)

    print(result)
    file_path = r'C:\Users\kawka\PycharmProjects\pythonProject\sap\SAP_SuccessFactors_Reviews.json'
    with open(file_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    return result


def print_reviews(reviews_data):
    if reviews_data:
        """Print reviews in a readable format."""
        for i, review in enumerate(reviews_data, start=1):
            print(f"Review {i}:")
            print(f"  - Name: {review['name']}")
            print(f"  - Title: {review['title']}")
            print(f"  - Pros:")
            for pro in review['pros']:
                print(f"    - {pro}")
            print(f"  - Cons:")
            for con in review['cons']:
                print(f"    - {con}")
            print()
    else:
        print("No reviews found.")


# Click "Load more reviews" multiple times to get more data
for _ in range(10):  # Adjust the range for more clicks
    try:
        load_more_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Load more reviews')]")
        print("Found 'Load more reviews' button. Clicking...")
        driver.execute_script("arguments[0].click();", load_more_button)
        time.sleep(5)  # Wait for reviews to load

        # Extract and print new reviews
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        new_reviews_data = extract_reviews(soup)
        #print_reviews(new_reviews_data)
    except Exception as e:
        print(f"Error: {e}")
        break  # Stop if the button is not found or no more reviews to load

driver.quit()
