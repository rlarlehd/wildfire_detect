
import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import cv2

model_path = "./wildfire_resnet18.pth"
class_names = ["fire", "nofire"]
image_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

features = None
grads = None

def save_features_hook(module, input, output):
    global features
    features = output

def save_grads_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0]

model.layer4.register_forward_hook(save_features_hook)
model.layer4.register_backward_hook(save_grads_hook)

def predict_image_with_cam(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)  

    _, pred = torch.max(output, 1)
    prob = torch.softmax(output, dim=1)[0][pred.item()].item()
    predicted_class = class_names[pred.item()]
    cam = None
    if predicted_class == "fire":
        model.zero_grad()
        score = output[0, pred.item()]
        score.backward()

        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        cam = torch.zeros(features.shape[2:], dtype=torch.float32).to(device)

        for i, w in enumerate(pooled_grads):
            cam += w * features[0, i, :, :]

        cam = cam.cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
    return predicted_class, prob, cam

def visualize(img, predicted_class, prob, cam):
    label = f"{predicted_class.upper()} ({prob*100:.1f}%)"
    color = (0, 0, 255) if predicted_class == "fire" else (0, 255, 0)
    img = cv2.putText(img, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
    if cam is not None:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        return overlay
    return img

if __name__ == "__main__":
    video_list = ["fire_1.mp4"]#, "fire_2.mp4", "fire_3.mp4", "fire_4.mp4"]
    for video in video_list:
        print(f"작업시작: {video}")
        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = os.path.splitext(os.path.basename(video))[0] + "_out.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("작업완료")
                break
            cls, p, cam = predict_image_with_cam(frame)
            vis = visualize(frame, cls, p, cam)
            out.write(vis)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
