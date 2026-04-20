"""
Run this ONCE to generate:
  - stress_cnn_model.pth
  - nlp_model.pkl
  - vectorizer.pkl

Just run:  python generate_voice_models.py
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 55)
print("  Generating Voice Stress Models")
print("=" * 55)


# ─────────────────────────────────────────────────────────
# 1. CNN MODEL  (trained on synthetic MFCC-like data)
# ─────────────────────────────────────────────────────────
print("\n[1/3] Training CNN audio model...")

class StressCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool2d((10, 10))
        self.fc1     = nn.Linear(32 * 10 * 10, 64)
        self.fc2     = nn.Linear(64, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def make_synthetic_mfcc(n=2000, random_state=42):
    """
    Simulate MFCC features (40 x 100) for stressed vs not-stressed.
    Stressed audio tends to have:
      - higher energy in lower MFCCs (coefficients 0-5)
      - more variance
      - faster tempo (higher values in time dimension)
    """
    np.random.seed(random_state)
    X, y = [], []
    for i in range(n):
        label = np.random.randint(0, 2)
        # Base noise
        mfcc = np.random.randn(40, 100).astype(np.float32)
        if label == 1:  # stressed
            mfcc[:6, :]  += np.random.uniform(1.5, 3.0)   # higher low-freq energy
            mfcc         *= np.random.uniform(1.2, 1.8)    # more variance
            mfcc[:, :50] += np.random.uniform(0.5, 1.0)    # faster onset
        else:           # calm
            mfcc[:6, :]  += np.random.uniform(0.0, 0.5)
            mfcc         *= np.random.uniform(0.7, 1.0)
        X.append(mfcc)
        y.append(label)
    return np.array(X), np.array(y)

X, y = make_synthetic_mfcc(n=3000)
X_t  = torch.tensor(X).unsqueeze(1)          # (N, 1, 40, 100)
y_t  = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
ds_train = TensorDataset(X_train, y_train)
loader   = DataLoader(ds_train, batch_size=32, shuffle=True)

cnn   = StressCNN()
opt   = optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    cnn.train()
    total_loss = 0
    for xb, yb in loader:
        opt.zero_grad()
        out  = cnn(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/15  loss={total_loss/len(loader):.4f}")

cnn.eval()
with torch.no_grad():
    preds = cnn(X_test).argmax(dim=1)
acc = (preds == y_test).float().mean().item()
print(f"  CNN Test Accuracy: {acc*100:.1f}%")

torch.save(cnn.state_dict(), "stress_cnn_model.pth")
print("  ✅ Saved: stress_cnn_model.pth")


# ─────────────────────────────────────────────────────────
# 2. NLP MODEL + VECTORIZER  (trained on synthetic sentences)
# ─────────────────────────────────────────────────────────
print("\n[2/3] Training NLP text model...")

STRESSED_SENTENCES = [
    "I feel so overwhelmed right now",
    "Everything is going wrong today",
    "I can't handle this anymore",
    "I am exhausted and burnt out",
    "I feel like I'm going to break down",
    "Nothing is working out for me",
    "I have too much on my plate",
    "I feel anxious and nervous all the time",
    "My head is pounding from all this pressure",
    "I haven't slept well in days",
    "I feel hopeless and lost",
    "There is so much pressure on me",
    "I feel like crying for no reason",
    "I am constantly worried about everything",
    "My heart is racing and I feel tense",
    "I feel so stressed out today",
    "I am struggling to focus on anything",
    "Everything feels too much to handle",
    "I keep making mistakes and feel terrible",
    "I feel really down and depressed",
    "I am under a lot of stress lately",
    "I feel irritated and angry all the time",
    "My anxiety is through the roof",
    "I feel disconnected from everyone around me",
    "I just want everything to stop",
    "Work is killing me with deadlines",
    "I feel panicked and cannot calm down",
    "I feel mentally drained and tired",
    "I am overwhelmed with responsibilities",
    "Nothing makes me happy anymore",
    "I feel like I am failing at everything",
    "I feel tense and wound up",
    "My mind won't stop racing",
    "I feel like giving up",
    "I am so stressed I cannot eat",
    "Everything feels like a disaster today",
    "I feel like I have no energy at all",
    "I am terrified of what might happen",
    "I feel so much pressure from all sides",
    "I cannot stop thinking about my problems",
]

CALM_SENTENCES = [
    "I feel relaxed and at peace today",
    "Everything is going well in my life",
    "I had a great sleep last night",
    "I feel calm and collected",
    "Life is good and I am happy",
    "I feel refreshed and energetic",
    "I am enjoying my day so much",
    "I feel content and satisfied",
    "I had a wonderful time with family",
    "I feel motivated and ready to work",
    "Everything is under control today",
    "I feel positive about the future",
    "I am grateful for everything I have",
    "I feel joyful and light today",
    "My mind is clear and focused",
    "I feel balanced and centered",
    "I had a peaceful and quiet morning",
    "I feel confident about my decisions",
    "I am in a great mood today",
    "I feel comfortable and secure",
    "Things are going smoothly for me",
    "I feel hopeful and optimistic",
    "I had a productive and fulfilling day",
    "I feel healthy and full of energy",
    "I am looking forward to the weekend",
    "I feel appreciated by the people around me",
    "I had a delicious meal and feel great",
    "I feel creative and inspired today",
    "I am enjoying the simple things in life",
    "I feel at ease with everything around me",
    "My work is progressing well",
    "I feel supported by my friends and family",
    "I am happy with how things turned out",
    "I feel lively and enthusiastic",
    "I had fun today and feel wonderful",
    "I feel satisfied with my achievements",
    "I am relaxed and enjoying the moment",
    "I feel free and unburdened",
    "I am in harmony with myself",
    "I feel cheerful and full of life",
]

# Augment with slight variations
import random
random.seed(42)
all_texts, all_labels = [], []
for _ in range(25):
    for s in STRESSED_SENTENCES:
        words = s.split()
        random.shuffle(words[:3])
        all_texts.append(" ".join(words))
        all_labels.append(1)
    for s in CALM_SENTENCES:
        words = s.split()
        random.shuffle(words[:3])
        all_texts.append(" ".join(words))
        all_labels.append(0)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_nlp = vectorizer.fit_transform(all_texts)
y_nlp = np.array(all_labels)

X_tr, X_te, y_tr, y_te = train_test_split(X_nlp, y_nlp, test_size=0.2, random_state=42)
nlp = LogisticRegression(max_iter=500, random_state=42)
nlp.fit(X_tr, y_tr)
acc_nlp = accuracy_score(y_te, nlp.predict(X_te))
print(f"  NLP Test Accuracy: {acc_nlp*100:.1f}%")

pickle.dump(nlp,        open("nlp_model.pkl",  "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("  ✅ Saved: nlp_model.pkl")
print("  ✅ Saved: vectorizer.pkl")


# ─────────────────────────────────────────────────────────
# 3. VERIFY ALL FILES EXIST
# ─────────────────────────────────────────────────────────
print("\n[3/3] Verifying files...")
import os
files = ["stress_cnn_model.pth", "nlp_model.pkl", "vectorizer.pkl"]
all_ok = True
for f in files:
    exists = os.path.exists(f)
    print(f"  {'✅' if exists else '❌'} {f}")
    if not exists:
        all_ok = False

print("\n" + "=" * 55)
if all_ok:
    print("  All models generated successfully!")
    print("  Now run:  python app.py")
    print("  Voice tab will be fully enabled ✅")
else:
    print("  ❌ Some files missing — re-run this script")
print("=" * 55)