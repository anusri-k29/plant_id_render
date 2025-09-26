from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from safetensors.torch import load_file
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# --- Load your model once at startup ---
# Replace with your actual CNN architecture!
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

model = SimpleCNN(num_classes=5)
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def root():
    return {"message": "Plant ID API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return JSONResponse({"prediction": int(prediction)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
