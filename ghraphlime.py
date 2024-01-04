import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import seaborn as sn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

def generate_perturbed_samples(data, num_samples=100, noise_level=0.01):
    perturbed_samples = []
    for index, row in data.iterrows():
        for _ in range(num_samples):
            noise = np.random.normal(0, noise_level, size=row.shape)
            new_sample = row + noise
            perturbed_samples.append(new_sample)

    return pd.DataFrame(perturbed_samples)

df_features_only = df.iloc[:, :-1]
df_perturbed = generate_perturbed_samples(df_features_only)

def graphlime_explanation(instance, perturbed_data, model, local_model=LinearRegression()):
    predictions = model.predict(perturbed_data)
    local_model.fit(perturbed_data, predictions)
    return local_model.coef_

for index, instance in df_features_only.iterrows():
  instance = df_features_only.iloc[index]
  perturbed_samples = generate_perturbed_samples(pd.DataFrame([instance]))

  explanation = graphlime_explanation(instance, perturbed_samples, model)
  print(f"Explication for instance {index}: {explanation}")

def plot_graphlime_explanation(coeficients, feature_names):
    coeficients = np.array(coeficients).flatten()
    if len(coeficients) != len(feature_names):
        raise ValueError("The len of coeficients is not equal with len of feature names")

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, coeficients, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Importance of features')
    plt.title('GraphLIME - Importance of features')
    plt.xticks(rotation=45)
    plt.show()

feature_names = df.columns[:-1]
coeficients = explanation
plot_graphlime_explanation(coeficients, feature_names)

coeficients = np.array(coeficients).flatten()
significant_features = [feature_names[i] for i, coef in enumerate(coeficients) if coef > 0]
print(significant_features)

df_filtered = df[significant_features]

df_filtered
X_new = df_filtered.copy()

df_filtered_with_outcome = df_filtered.copy()
df_filtered_with_outcome['Outcome'] = df['Outcome']

df_filtered_with_outcome

df_new = df_filtered_with_outcome
y = df_new['Outcome']
X_train_nou, X_test_nou, y_train_nou, y_test_nou = train_test_split(X_new, y, test_size=0.2, random_state=0)
model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=([df_new.shape[1] - 1])),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_nou = np.asarray(X_train_nou).astype(np.float32)
y_train_nou = np.asarray(y_train_nou).astype(np.float32)

X_test_nou = np.asarray(X_test_nou).astype(np.float32)
y_test_nou = np.asarray(y_test_nou).astype(np.float32)

model.fit(X_train_nou, y_train_nou, epochs=10, batch_size=60, validation_data=(X_test_nou, y_test_nou))

y_pred_nou = model.predict(X_test_nou)
y_pred_nou = y_pred_nou > 0.45

np.set_printoptions()
cm = confusion_matrix(y_test_nou, y_pred_nou)
ac = accuracy_score(y_test_nou, y_pred_nou)

cm
labels = [0, 1]
df_cm = pd.DataFrame(cm, labels, labels)
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, square=True, cbar=False, fmt='g')
ax.set_ylim(0, 2)
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.invert_yaxis()
plt.show()

fpr_nou, tpr_nou, _ = roc_curve(y_test_nou, y_pred_nou)
roc_auc_nou = auc(fpr_nou, tpr_nou)

plt.figure(figsize=(10, 6))
plt.plot(fpr_nou, tpr_nou, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nou)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for our model')
plt.legend(loc="lower right")
plt.show()
