from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
from PIL import Image
import json
from llm_agent import generate_advice

app = Flask(__name__)

DEVICE = torch.device("cpu")

# Load ontology
with open("ontology/agriculture_ontology.json") as f:
    ontology = json.load(f)

# Load model
NUM_CLASSES = len(ontology.keys())
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load("models/mobilenet_model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = list(ontology.keys())

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    advice = None
    meta = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        prediction = class_names[pred.item()]

        #advice = generate_advice(prediction)
        #meta = ontology.get(prediction, {})
        meta = ontology.get(prediction, {})

        category = meta.get("category", "Unknown")
        treatment = meta.get("treatment", "Consult agricultural expert.")

        advice = generate_advice(prediction, category, treatment)


    return render_template("index.html", prediction=prediction, advice=advice, meta=meta)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True, use_reloader=False)