from acm_submission import Model_Skin
import torch
from torch.utils.data import DataLoader , Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
from torchvision import transforms
import os


image_folder = "HAM10000_images_part_1/"

df = pd.read_csv("HAM10000_metadata.csv")
df = df.drop(["lesion_id"] , axis=1)


image_files = set([fname.split('.')[0] for fname in os.listdir(image_folder) if fname.endswith('.jpg')])

df = df[df["image_id"].isin(image_files)]

df["image_path"] = df["image_id"].apply(lambda x: os.path.join(image_folder , f"{x}.jpg"))

label_encoder = LabelEncoder()
label_encoder.fit(df["dx"])
df["dx_encoded"] = label_encoder.transform(df["dx"])

print(len(df["dx_encoded"].unique()))

img_size = 64

class Dataset_Prep(Dataset):
    def __init__(self , dataframe:pd.DataFrame , transform = None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image at {row['image_path']}: {e}")
            # Return a blank image and label 0 to avoid crashing
            image = Image.new("RGB", (img_size, img_size))
            label = 0
        else:
            label = row["dx_encoded"]

        if self.transform:
            image = self.transform(image)

        return image , torch.tensor(label)


transform = transforms.Compose([
    transforms.Resize((img_size , img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
])

dataset = Dataset_Prep(dataframe=df , transform=transform)
train_size = int(len(dataset)*0.8)

if(len(dataset) == len(image_files)):
    print("data prepration")
    train_dataset , test_dataset = torch.utils.data.random_split(dataset , [train_size, int(len(dataset) - train_size)])
    train_loader = DataLoader(train_dataset , 32 , True )
    test_loader = DataLoader(test_dataset , 32 , False )



print("Instantiating model...")
model = Model_Skin(128 , len(df["dx_encoded"].unique()) , img_size)

epochs = 50
optimizer = torch.optim.Adam(model.parameters() , 3e-4 , weight_decay=0.04)
loss_fn = torch.nn.CrossEntropyLoss()
losses = []

for epoch in range(epochs):
    model.train()
    for batch_idx, (image , label) in enumerate(train_loader):
        try:
            output = model(image)
            loss = loss_fn(output , label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        except Exception as e:
            print(f"Error in training loop at epoch {epoch+1}, batch {batch_idx}: {e}")
            break

    if(epoch%10==0):
        print(f"Epoch: {epoch+1} || Last batch loss: {losses[-1]}")

import matplotlib.pyplot as plt
plt.plot(losses)

def accuracy(true_labels , pred_labels):
    preds = torch.argmax(pred_labels , dim=1)
    correct = (preds == true_labels).sum().item()
    total = true_labels.size(0)
    return correct/total

accu = []
print("Starting evaluation loop...")
for batch_idx, (image,label) in enumerate(test_loader):
    try:
        output = model(image)
        loss = loss_fn(output , label)

        accurate = accuracy(label , output)
        accu.append(accurate)
        if batch_idx % 10 == 0:
            print(f"Test Batch {batch_idx}: Accuracy {accurate}")
    except Exception as e:
        print(f"Error in evaluation loop at batch {batch_idx}: {e}")
        break

plt.plot(accu)
print("Script finished.")

torch.save(model.state_dict() , "model.pth")
print("model is saved")
