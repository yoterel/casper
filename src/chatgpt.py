import json
from pathlib import Path
import os  # Included to Python
from openai import OpenAI


class ChatGPTClient:
    def __init__(self):
        apikey = os.getenv("OPENAI_API_KEY")
        if apikey:
            print("Found API key: {}".format(apikey))
            self.client = OpenAI(api_key=apikey)
        else:
            raise Exception("API key not found")

    def send_request(self, encoded_image: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "output must follow this format {'animals': ['animal1', 'animal2', 'animal3']}. This is a picture of a hand gesture. Which animal is it most similar to? Return 3 animals by priority in the requested format.",
                        },
                        {
                            "type": "image_url",
                            "image_url": encoded_image,
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        try:
            print("raw response: ", response.choices[0].message.content)
            response = self.decode_response(response)["animals"][0]
        except Exception as e:
            print("Error in decoding response!")
            print(e)
            response = ""

        return response

    def decode_response(self, response) -> dict:
        return eval(response.choices[0].message.content)

    def extract_animal(self, response) -> str:
        try:
            animal = json.loads(
                response.choices[0].message.content.replace("json", "").replace("`", "")
            )["animals"][0]
        except:
            animal = None
        return animal


def encode_image(image_path: Path):
    import cv2
    import base64

    img = cv2.imread(image_path.as_posix())
    retval, bytes = cv2.imencode(".png", img)
    encoded_image = f"data:image/jpeg;base64,{base64.b64encode(bytes).decode('utf-8')}"
    return encoded_image


if __name__ == "__main__":
    encoded_image = encode_image(Path("../../resource/images/mask.png"))
    client = ChatGPTClient()
    response = client.send_request(encoded_image)
    print(response)
