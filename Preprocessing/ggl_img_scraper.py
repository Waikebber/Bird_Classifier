import os, time, requests

def google_image_download(query, save_directory, api_key, cx, n = 10):
    # Create the Google Custom Search API URL
    url = f"https://www.googleapis.com/customsearch/v1?searchType=image&key={api_key}&cx={cx}&q={query}"

    # Set the User-Agent header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    }

    # Send a GET request to the API URL with the User-Agent header
    while True:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            break  # Exit the loop if the request is successful
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            if response.status_code == 429:
                # Handle rate limit exceeded error by waiting for a minute
                print("Rate limit exceeded. Retrying in 1 minute...")
                time.sleep(60)
            else:
                return  # Exit the function if an error occurs and cannot be resolved

    # Get the JSON response data
    json_data = response.json()

    # Extract the image URLs from the JSON response
    image_urls = [item['link'] for item in json_data['items']]

    # Save the specified number of images to the specified directory
    os.makedirs(save_directory, exist_ok=True)
    saved_images = 0
    for i, image_url in enumerate(image_urls):
        if saved_images >= n:
            break
        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            file_path = os.path.join(save_directory, f"image{saved_images+1}.jpg")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            saved_images += 1
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            if response.status_code == 429:
                wait_time = 2 ** saved_images  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                break