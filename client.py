import requests

# Replace with your deployed API URL
url = 'https://your-app.onrender.com/predict'

# Path(s) to example image file(s) you want to test
image_paths = [
    'path/to/image1.jpg',
    'path/to/image2.png',
    # Add more image file paths as needed
]

predictions = []

for img_path in image_paths:
    with open(img_path, 'rb') as f:
        files = {'file': (img_path, f, 'image/jpeg')}  # Adjust MIME type as needed
        response = requests.post(url, files=files)

    if response.ok:
        result = response.json()
        predictions.append(result.get('predicted_class', 'No class returned'))
    else:
        predictions.append(f"Error: {response.status_code}")

print(predictions)
