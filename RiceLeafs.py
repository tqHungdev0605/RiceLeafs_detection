import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách lớp bệnh
classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

# Tải mô hình ResNet50
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transform ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7831, 0.7906, 0.7777], std=[0.1815, 0.1717, 0.2786])
])

# Hàm dự đoán
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = classes[pred_idx]
        pred_prob = probs[pred_idx].item() * 100
    return pred_class, pred_prob

# Giao diện Tkinter
class RiceDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện bệnh lá lúa")
        self.root.geometry("1200x800")
        self.root.configure(bg="#d2a679")  # Màu nền nâu nhẹ
        
        self.model = load_model('resnest50_0.85.pt')
        self.image_paths = []
        
        # Frame tiêu đề
        self.title_frame = tk.Frame(root, bg="#8b5a2b", pady=15)
        self.title_frame.pack(fill="x")
        self.title_label = tk.Label(self.title_frame, text="Nhận diện bệnh lá lúa", font=("Helvetica", 20, "bold"), fg="white", bg="#8b5a2b")
        self.title_label.pack()
        
        # Frame điều khiển
        self.control_frame = tk.Frame(root, bg="#d2a679", pady=10)
        self.control_frame.pack()
        
        self.select_button = tk.Button(self.control_frame, text="Chọn ảnh", command=self.select_images, font=("Helvetica", 12, "bold"), bg="#ff6600", fg="white", padx=20, pady=10)
        self.select_button.grid(row=0, column=0, padx=20)
        
        self.predict_button = tk.Button(self.control_frame, text="Dự đoán", command=self.predict_images, font=("Helvetica", 12, "bold"), bg="#ff0000", fg="white", padx=20, pady=10)
        self.predict_button.grid(row=0, column=1, padx=20)
        
        # Frame hiển thị ảnh
        self.image_frame = tk.Frame(root, bg="#ffffff", pady=10, relief="groove", borderwidth=2)
        self.image_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        # Thanh trạng thái
        self.status_frame = tk.Frame(root, bg="#cbb092", pady=5)
        self.status_frame.pack(fill="x", side="bottom")
        self.status_label = tk.Label(self.status_frame, text="Số ảnh đã chọn: 0", font=("Helvetica", 10), bg="#cbb092", fg="#333333")
        self.status_label.pack()
        
        # Danh sách label
        self.image_labels = []
        self.result_labels = []
    
    def select_images(self):
        self.clear_images()
        files = filedialog.askopenfilenames(title="Chọn ảnh lá lúa", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.image_paths = list(files)[:10]
            for i, path in enumerate(self.image_paths):
                img = Image.open(path).convert('RGB').resize((200, 200))
                photo = ImageTk.PhotoImage(img)
                img_label = tk.Label(self.image_frame, image=photo, bg="#ffffff", relief="raised", borderwidth=2)
                img_label.image = photo
                img_label.grid(row=i//5 * 2, column=i%5, padx=10, pady=10)
                img_label.bind("<Enter>", lambda e, lbl=img_label: lbl.config(relief="sunken"))
                img_label.bind("<Leave>", lambda e, lbl=img_label: lbl.config(relief="raised"))
                self.image_labels.append(img_label)
                result_label = tk.Label(self.image_frame, text="Chưa dự đoán", font=("Helvetica", 11), bg="#ffffff", fg="#666666")
                result_label.grid(row=i//5 * 2 + 1, column=i%5, padx=10, pady=5)
                self.result_labels.append(result_label)
            self.status_label.config(text=f"Số ảnh đã chọn: {len(self.image_paths)}")
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ít nhất 1 ảnh!")
    
    def predict_images(self):
        if not self.image_paths:
            messagebox.showerror("Lỗi", "Chưa chọn ảnh nào!")
            return
        self.status_label.config(text="Đang dự đoán...")
        self.root.update()
        for i, path in enumerate(self.image_paths):
            pred_class, pred_prob = predict_image(self.model, path)
            result_text = f"{pred_class}: {pred_prob:.2f}%"
            color = "#008000" if pred_class == "Healthy" else "#ff0000"
            self.result_labels[i].config(text=result_text, fg=color, font=("Helvetica", 11, "bold"))
        self.status_label.config(text=f"Dự đoán hoàn tất! Số ảnh: {len(self.image_paths)}")
    
    def clear_images(self):
        for label in self.image_labels + self.result_labels:
            label.destroy()
        self.image_labels.clear()
        self.result_labels.clear()
        self.image_paths.clear()
        self.status_label.config(text="Số ảnh đã chọn: 0")

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = RiceDiseaseApp(root)
    root.mainloop()
