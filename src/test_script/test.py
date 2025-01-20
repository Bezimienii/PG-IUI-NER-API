import time
import requests
from constants import MODELS, BASE_URL, HEADERS, TRAIN


def get_model_id_by_model_name(model_name: str):
    url = f"{BASE_URL}/{MODELS}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        models = data.get("models", [])

        # Find ID by base model name
        model_id = None
        model_path = "models/" + model_name

        for model in models:
            if model.get("file_path") == model_path:
                model_id = model.get("id")

        return model_id
    else:
        print("Failed to fetch models. Status Code:", response.status_code)
        return None

def test_model(model_id: int, input_text: str):
    url = f"{BASE_URL}/{MODELS}/{model_id}/ner"
    payload = {
        "input_text": input_text
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    print(f"Test Model {model_id} Response:", response.status_code, response.json())


def train_model(path: str, model_name: str):
    url = f"{BASE_URL}/{MODELS}/{TRAIN}"

    files = {
        "train_data": open(f"{path}/train_sample.conllu", "rb"),
        "valid_data": open(f"{path}/val_sample.conllu", "rb"),
        "test_data": open(f"{path}/test_sample.conllu", "rb")
    }
    data = {
        "model_name": model_name,
        "base_model": "1"
    }
    
    response = requests.post(url, data=data, files=files)
    if response.status_code == 200:
        print("Process model response:", response.json())
    else:
        print("Failed to process model. Status Code:", response.status_code, "Response:", response.text)


def check_status(model_id):
    url = f"{BASE_URL}/{MODELS}/{model_id}/state"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        is_training = data.get("is_training", "")
        is_trained = data.get("is_trained", "")

        return is_training, is_trained


def check_model_trained(model_id):
    is_training=True
    while(is_training):
        is_training, is_trained = check_status(model_id)
        print("still training model")
        
        if not is_training:
            break
    
        time.sleep(60)

    print(is_training, is_trained)

def main():
    print("Running Training API endpoints tests\n")
    
    try:
        eng_model_id = 1
        pl_model_id = 2
        test_model(eng_model_id, "The Oracle Solution is the one thing I love.")
        test_model(pl_model_id, "Nowy Jork to fajne miasto ale i tak bardziej lubię Isaaca El Newtona.")

        train_model("en_sample", "final_test_en")
        test_model_id = get_model_id_by_model_name("final_test_en")

        check_model_trained(test_model_id)
        
        test_model(test_model_id, "Oracle is good.")

        train_model("pl_sample", "final_test_pl")
        test_model_id = get_model_id_by_model_name("final_test_pl")

        check_model_trained(test_model_id)
        
        test_model(test_model_id, "Warszawa jest ok, ale Gdańsk jest super - Jan Kowalski")

    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
