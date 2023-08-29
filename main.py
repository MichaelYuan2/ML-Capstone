import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
random.seed(14669588)
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']


def make_dataloader(dataframe, group_name):
    grouped = dataframe.groupby(group_name)
    test_indices = []

    for name, group in grouped:
        test_index = group.sample(n=500, random_state=42).index.values
        test_indices.extend(test_index)

    train = dataframe.drop(test_indices)
    test = dataframe.loc[test_indices]
    X_train, y_train = train.drop(['music_genre'], axis=1), train['music_genre']
    X_val, y_val = test.drop(['music_genre'], axis=1), test['music_genre']
    X_train, X_val, y_train, y_val = X_train.to_numpy(), X_val.to_numpy(), y_train.to_numpy(), y_val.to_numpy()
    X_train, X_val = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
    y_train, y_val = torch.LongTensor(y_train), torch.LongTensor(y_val)
    y_train, y_val = torch.reshape(y_train, (-1,)), torch.reshape(y_val, (-1,))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_val, y_val)
    test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=True)

    return train_loader, test_loader


def plot_auc(y_true, y_prob, model_name):
    # Transform target into binary form
    lb = LabelBinarizer().fit(y_true)
    y_onehot_test = lb.transform(y_true)
    # print(y_onehot_test.shape)

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    n_classes = 10

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.3f}")

    # Compute macro-average ROC curve and ROC area
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.3f}")

    # Plot all the graph together
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.3f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_prob[:, class_id],
            name=f"ROC curve for {class_id}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for model: {}".format(model_name))
    plt.legend()
    plt.show()

    return roc_auc["macro"]


# Read in the data
df = pd.read_csv('musicData.csv')
print(df.info())
print(df.head())
# print(df.value_counts())

#df.hist(bins=50, figsize=(20, 20))
#plt.show()

### Data Processing
# Get rid of unnecessary info
df = df.drop(['instance_id', 'obtained_date', 'artist_name', 'track_name'], axis=1)

# Handling miss values
df = df.dropna(subset=['music_genre'])

print((df['duration_ms'] == -1).sum())
print((df['tempo'] == "?").sum())

mask = df['duration_ms'] != -1
group_means = df[mask].groupby('music_genre')['duration_ms'].mean()
df.loc[~mask, 'duration_ms'] = df.loc[~mask, 'music_genre'].apply(lambda x: group_means[x])

df['tempo'] = df['tempo'].replace("?", -1)
df = df.astype({'tempo': 'float64'})
mask = df['tempo'] != -1
group_means = df[mask].groupby('music_genre')['tempo'].mean()
df.loc[~mask, 'tempo'] = df.loc[~mask, 'music_genre'].apply(lambda x: group_means[x])

# Encode categorical data
le = LabelEncoder().fit(df['music_genre'].unique())
df['music_genre'] = le.transform(df['music_genre'])

dummy_data = ['key', 'mode']
for col in dummy_data:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    df = df.drop(col, axis=1)
df = df.drop(['key_A', 'mode_Major'], axis=1)

# Normalize Data
norm_cols = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
             'instrumentalness', 'liveness', 'loudness', 'tempo', 'speechiness', 'valence']
for col in norm_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

print(df.info())


# Make train and test loader
train_loader, test_loader = make_dataloader(df, 'music_genre')
train, test = train_test_split(df, test_size=5000, stratify=df['music_genre'])
X_train, y_train = train.drop('music_genre', axis=1), train['music_genre']
X_val, y_val = test.drop('music_genre', axis=1), test['music_genre']


### Define DNN
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Train and test
def train(model, dataloader, criterion, optimizer, epoch=0):
    model.train()
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        # data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    return epoch_loss / len(dataloader)


def test(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        data, target = next(iter(dataloader))
        output = model(data)

        pred = output.argmax(dim=1)
        test_loss = criterion(output, target)

        y_true = target.cpu().numpy()
        # print(y_true.shape)
        y_prob = torch.softmax(output, dim=1).cpu().numpy()
        # print(y_prob.shape)

        correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()
        accuracy = 100. * correct / len(dataloader.dataset)
        print('\nTest set: CrossEntropy: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss/len(dataloader), correct, len(dataloader.dataset),
            accuracy))

    plot_auc(y_true, y_prob, 'DNN')

    return 0


# Main Function
def dnn():
    learning_rate = 1e-3
    num_epochs = 100
    input_size = 23
    output_size = 10

    mlp_classifier = MLP(input_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_classifier.parameters(), lr=learning_rate)
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = train(mlp_classifier, train_loader, epoch=epoch, criterion=criterion, optimizer=optimizer)
        train_losses.append(epoch_loss.detach().numpy())
        if epoch % 10 == 0:
            print("Epoch: {}/{}, training loss: {}".format(epoch, num_epochs, epoch_loss))

    plt.plot(train_losses)
    plt.title('Training Loss of general model with Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()

    test(mlp_classifier, test_loader, criterion=criterion)


# t_SNE
def t_sne():
    X = df.drop('music_genre', axis=1)
    y = df['music_genre']
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_results = tsne.fit_transform(X)

    # Plot the t-SNE results
    plt.figure(figsize=(30, 20))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='tab10')
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("t-SNE plot of 2 dimensions")
    plt.colorbar()
    plt.savefig("t-SNE_plot")
    plt.show()

    tsne_results = pd.DataFrame(tsne_results, columns=['Dimension1', 'Dimension2'])
    tsne_results['music_genre'] = y
    print(tsne_results.head())
    print(tsne_results.isnull().sum())
    tsne_results.dropna(subset=['music_genre'])
    # np.savetxt('X_after_t-SNE_2.csv', tsne_results, delimiter=',')

    return tsne_results


# PCA
def pca():
    # PCA
    X = df.drop('music_genre', axis=1)
    y = df['music_genre']
    pca_clf = PCA(n_components=2, whiten=True).fit(X)
    X_pca = pca_clf.transform(X)

    plt.figure(figsize=(30, 20))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10')
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("PCA plot of 2 dimensions")
    plt.colorbar()
    plt.savefig("PCA_plot")
    plt.show()

    C = X.corr()
    eigvals, eigvecs = np.linalg.eig(C)
    print(eigvals)


# Random Forest
def random_forest(X_train, y_train, X_val, y_val):
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced',
                                    max_samples=0.2, bootstrap=True, criterion='gini') \
        .fit(X_train, y_train)
    rf_prob = rf_clf.predict_proba(X_val)

    roc_auc = plot_auc(y_val, rf_prob, 'Random Forest')
    return roc_auc


def adaboost(X_train, y_train, X_val, y_val):
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=10, class_weight='balanced'),
                                 n_estimators=100, learning_rate=1).fit(X_train, y_train)
    ada_prob = ada_clf.predict_proba(X_val)

    plot_auc(y_val, ada_prob, 'AdaBoost')
    return 0


def k_means(X_tsne):
    # K-means with k=10
    kmeans = KMeans(10, n_init='auto')
    labels = kmeans.fit_predict(X_tsne)
    print("The total sum of distance of K-means is: {}".format(kmeans.fit(X_tsne).inertia_))

    color_map = {}
    for i in range(10):
        color_map[i] = colors[i]
    color = [color_map[label] for label in labels]

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('K-means Clustering with k=10 on Music Genre')
    plt.show()


# Main Program
# pca()
# full_roc_auc = random_forest(X_train, y_train, X_val, y_val)
# adaboost(X_train, y_train, X_val, y_val)
# dnn()

# df_tsne = t_sne()

df_tsne = pd.read_csv('X_after_t-SNE_2.csv')
df_tsne.columns = ['Dimension 1', 'Dimension 2', 'music_genre']
df_tsne = df_tsne.dropna(subset=['music_genre'])
print(df_tsne['music_genre'].isnull().sum())
tsne_train, tsne_val = train_test_split(df_tsne, test_size=5000, stratify=df_tsne['music_genre'])
X_train_tsne, y_train_tsne = tsne_train.drop('music_genre', axis=1), tsne_train['music_genre']
X_val_tsne, y_val_tsne = tsne_val.drop('music_genre', axis=1), tsne_val['music_genre']
# random_forest(X_train_tsne, y_train_tsne, X_val_tsne, y_val_tsne)
k_means(df_tsne.drop('music_genre', axis=1).to_numpy())

'''
# test for key features
min_roc_auc = full_roc_auc
key = 'None'
for col in X_train.columns:
    X_train_reduced, X_val_reduced = X_train.drop(columns=col), X_val.drop(columns=col)
    roc_auc_reduced = random_forest(X_train_reduced, y_train, X_val_reduced, y_val)
    if roc_auc_reduced < min_roc_auc:
        key = col
        min_roc_auc = roc_auc_reduced
print("The most significant factor is {}".format(key), min_roc_auc)
'''


# Extra Credit
popularity_dict = {}
for name, group in df.groupby('music_genre'):
    name = le.inverse_transform([name])
    popularity_dict[name[0]] = group['popularity'].mean()
    # print("Group: {}, mean_popularity: {}".format(name, group['popularity'].mean()))


# Extract keys and values from the dictionary
sorted_data = sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_data)
print(labels)

# Create histogram
plt.bar(labels, values, width=0.8)

# Add labels and title
plt.xlabel('Class Labels')
plt.ylabel('Values')
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Histogram by Class Labels')

plt.show()
