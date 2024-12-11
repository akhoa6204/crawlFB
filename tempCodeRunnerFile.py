import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import underthesea
import re   

# Kiểm tra thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Siêu tham số
input_size = 1000  # Kích thước từ vựng
hidden_size = 128
num_layers = 2
num_classes = 3  # Positive, Negative, Neutral
batch_size = 32
num_epochs = 5
learning_rate = 0.01

# Đọc và gắn nhãn dữ liệu train
def load_train_data():
    with open('positive.txt', 'r', encoding='utf-8') as f:
        positive_texts = f.readlines()
    with open('negative.txt', 'r', encoding='utf-8') as f:
        negative_texts = f.readlines()
    with open('neutral.txt', 'r', encoding='utf-8') as f:
        neutral_texts = f.readlines()

    texts = positive_texts + negative_texts + neutral_texts
    labels = [0] * len(positive_texts) + [1] * len(negative_texts) + [2] * len(neutral_texts)
    return texts, labels

# Xử lý văn bản
def text_preprocessing(doc):
    text_pre = doc.lower()
    
    text_pre = underthesea.word_tokenize(text_pre, format="text")
    text_pre = underthesea.text_normalize(text_pre)
    
    text_pre = text_pre.replace('XXXX', '')        
    text_pre = text_pre.replace(u'\ufffd', '')  # Replaces the ASCII symbol with ''
    text_pre = re.sub(r'[^\w\s]', '', text_pre)
    text_pre = text_pre.rstrip('\n')           # Removes line breaks
    text_pre = text_pre.casefold()             # Makes all letters lowercase
    
    text_pre = re.sub('\W_', ' ', text_pre)    # Removes special characters
    text_pre = re.sub("\S*\d\S*", " ", text_pre)  # Removes numbers and words concatenated with numbers
    text_pre = re.sub("\S*@\S*\s?", " ", text_pre)  # Removes emails and mentions
    text_pre = re.sub(r'http\S+', '', text_pre)  # Removes URLs with http
    text_pre = re.sub(r'www\S+', '', text_pre)   # Removes URLs with www
        
    # Remove stop words
    filename = "vietnamese-stopwords.txt"
    with open(filename, "r", encoding="utf-8") as f:
        List_StopWords = f.read().split("\n")
    text_pre = " ".join(text for text in text_pre.split() if text not in List_StopWords)
    
    return text_pre

# Load dữ liệu train
train_texts, train_labels = load_train_data()
train_texts = text_preprocessing(train_texts)

vectorizer = CountVectorizer(max_features=input_size)
X_train = vectorizer.fit_transform(train_texts).toarray()
vocab = vectorizer.get_feature_names_out()

# Dataset tùy chỉnh
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Tạo DataLoader
train_dataset = TextDataset(X_train, train_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Định nghĩa mô hình BiRNN
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Một lớp fully connected để dự đoán lớp đầu ra
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 vì BiRNN có 2 hướng (forward và backward)

    def forward(self, x):
        # LSTM trả về output và hidden states
        out, _ = self.lstm(x)
        
        # Lấy hidden state cuối cùng của tất cả các bước thời gian
        out = out[:, -1, :]
        
        # Pass qua lớp fully connected
        out = self.fc(out)
        return out

# Khởi tạo mô hình
model = BiRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)

# Định nghĩa loss và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Tính toán loss
        loss = criterion(outputs, labels)
        
        # Backward pass và cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
