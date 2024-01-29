import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Import joblib for saving the model

# Load the dataset from a CSV file
df = pd.read_csv("dataset.csv")

# Assuming you have columns named 'text' and 'label_num' in your CSV file
X = df['text']
y = df['label_num']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
    min_df=2,     # Ignore terms that appear in fewer than 2 documents
    sublinear_tf=True,  # Apply sublinear scaling to the TF
)

nb_classifier = MultinomialNB(alpha=0.1)  # Adjust the alpha (Laplace smoothing) parameter

model = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('nb', nb_classifier),
])


# Train the model
model.fit(X_train, y_train)

# Save the trained model as a .sav file
joblib.dump(model, 'naive_bayes_model.sav')

# Make predictions on the test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

with open('nb.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy:.2f}\n')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Calculate ROC curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]  # Get probability estimates for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
