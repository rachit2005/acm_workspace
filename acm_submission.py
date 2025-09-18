'''
Cases include a representative collection of all important diagnostic categories in the realm of 
pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), 
basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses 
and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) 
and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
'''


from torch import nn

class Model_Skin(nn.Module):
    def __init__(self, hidden_units, num_classes, img_size=128):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> 64x64
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> 32x32
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> 16x16
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_units*4 * (img_size//8) * (img_size//8), num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

