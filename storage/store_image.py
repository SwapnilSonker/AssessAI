import requests


def upload_to_cloudflare(image_path , api_key , account_id):
    print(f"inside the cloud function : {image_path} ,{api_key} , {account_id}")
    with open(image_path , "rb") as image_file:
        files = {'file' : image_file}
        headers = {
            'Authorization' : f"Bearer {api_key}"
        }
        url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/images/v1'
        response = requests.post(url, headers=headers, files=files)

        if requests.status_codes == 200:
            result = response.json()['result']
            image_url = result['variants'][0] 
            print("✅ Image Uploaded Successfully:")
            print("🌐 Image URL:", image_url)
            return image_url
        else:
            print("❌ Upload Failed:", response.json())
            return None
        
