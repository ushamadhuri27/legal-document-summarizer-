import pandas as pd
import random
import re
data1=pd.read_csv("CaseClassifyData.csv")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
divorce_phrases = [
    "I want a divorce from my spouse",
    "How do I start divorce proceedings?",
    "Looking for a lawyer to file for divorce",
    "My partner and I have decided to separate",
    "I want to legally end my marriage",
    "How can I file for divorce?",
    "Need help with divorce papers",
    "Want to separate from my husband",
    "Want to separate from my wife",
    "Filing for a divorce due to irreconcilable differences",
    "Can you guide me through the divorce process?",
    "I am not happy in my marriage, want to separate",
    "I want full custody after divorce",
    "Need legal advice for divorce settlement",
    "Me and my spouse mutually agreed to divorce",
    "How much does it cost to get divorced?",
    "Looking for an attorney for a divorce case",
    "Want to apply for a divorce from my wife",
    "Want to apply for a divorce from my husband",
    "Seeking help to dissolve my marriage",
    "I need assistance with divorce documentation",
    "What is the divorce procedure in my city?",
    "Need help to initiate divorce case",
    "Planning to separate from my spouse legally",
    "Going through a bad marriage, need legal separation",
    "How to serve divorce papers to my spouse?",
    "What documents are required for divorce?",
    "My marriage is falling apart; I need a divorce",
    "Getting separated from my wife soon, need advice",
    "Getting separated from my husband soon, need advice",
    "Looking to hire a divorce attorney",
    "Who can help me with a quick divorce?",
    "I need a lawyer for contested divorce",
    "I need a lawyer for mutual divorce",
    "Divorce case: child custody involved",
    "Legal support required for property division in divorce",
    "I’m stuck in an unhappy marriage, need legal help",
    "Can I get divorced without going to court?",
    "I want legal separation immediately",
    "I need to get out of a toxic marriage",
    "Planning to file divorce against my spouse",
    "Need guidance on contested divorce",
    "Mutual consent divorce: Need documentation support",
    "Want to end my marriage legally",
    "I’m separated already, want to file divorce formally",
    "Need legal consultation for divorce",
    "How long does a divorce case take?",
    "Filing divorce: my spouse doesn’t agree",
    "My husband is abusive, want divorce",
    "My wife cheated on me, filing for divorce",
    "Need alimony guidance after divorce",
    "Looking for affordable divorce lawyer",
    "Legal advice on filing divorce in another state",
    "Can I get alimony after separation?",
    "Spouse doesn’t cooperate for divorce, need help",
    "I want a fast divorce process",
    "Mutual separation, need help filing",
    "Want to get custody of children post-divorce",
    "Need help with divorce case hearing preparation",
    "Need help splitting assets after divorce",
    "Going through separation, need legal counsel",
    "Starting divorce proceedings, what’s the first step?",
    "Filing for divorce due to domestic violence",
    "Want to annul my marriage",
    "Looking for a family law attorney for divorce",
    "Need urgent divorce due to personal safety issues",
    "I want a divorce lawyer with experience in custody battles",
    "Can I file divorce online?",
    "Need divorce help: I’ve been abandoned by my spouse",
    "How to divide jointly owned property in divorce?",
    "Seeking legal help for child custody and divorce",
    "Husband has deserted me, want to file for divorce",
    "Need help with legal fees for divorce",
    "Can I get maintenance after divorce?",
    "Help with mutual divorce paperwork",
    "Can you guide me with divorce mediation?",
    "How to handle contested divorce cases?",
    "Need urgent legal help with spouse separation",
    "Marriage not working out, want legal separation",
    "How to claim alimony during divorce proceedings?",
    "What happens to my property in divorce?",
    "Looking for help with uncontested divorce",
    "Going through separation, worried about kids’ custody",
    "What are the rights of women in divorce cases?",
    "Need to file divorce due to financial disputes",
    "Filing divorce because of infidelity by spouse",
    "How to file divorce if spouse is abroad?",
    "My spouse is not cooperating for divorce, need help",
    "Looking for quick mutual consent divorce",
    "How to handle divorce property settlements?",
    "Want divorce lawyer for child custody matters",
    "I want to start my life fresh, need divorce help",
    "How long does contested divorce take?",
    "Can I get maintenance for children after divorce?",
    "Help me initiate my divorce case",
    "Want to separate formally from my spouse legally",
    "Need assistance filing for divorce online",
    "Can you help draft my divorce petition?",
    "How to claim maintenance for children after divorce?",
    "Mutual divorce: what are my rights?",
    "Need urgent appointment with divorce lawyer",
    "How to prove cruelty in divorce case?",
    "Looking for help in filing domestic violence and divorce case"
]

# Generate 100 random samples
extra_divorce_data = pd.DataFrame({
    'caseDescription': random.sample(divorce_phrases * 5, 100),  # repeat list to ensure enough samples
    'caseType': ['Divorce'] * 100
})

# Clean the descriptions (use your actual clean_text function)
extra_divorce_data['caseDescription'] = extra_divorce_data['caseDescription'].astype(str).apply(clean_text)

# Append to existing data
data1 = pd.concat([data1, extra_divorce_data], ignore_index=True)

# Optional: Preview
import pandas as pd
import numpy as np
import re
import joblib
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------
# Load and Clean Data
# ---------------------------
data2 = pd.read_csv("ClientClassifyData.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

data1['caseDescription'] = data1['caseDescription'].astype(str).apply(clean_text)
data2['caseDescription'] = data2['caseDescription'].astype(str).apply(clean_text)

x = data1['caseDescription'].tolist() + data2['caseDescription'].tolist()
y = data1['caseType'].tolist() + data2['caseType'].tolist()

print("Total samples:", len(x), len(y))

# ---------------------------
# Encode Labels
# ---------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'label_encoder.joblib')
n_classes = len(label_encoder.classes_)

# ---------------------------
# Tokenizer and Model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained(
    'nlpaueb/legal-bert-base-uncased', num_labels=n_classes
)

# ---------------------------
# Train-Validation Split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Tokenize
def tokenize_texts(texts, tokenizer, max_len=256):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )

train_encodings = tokenize_texts(X_train, tokenizer)
val_encodings = tokenize_texts(X_val, tokenizer)

train_labels_tf = tf.convert_to_tensor(y_train)
val_labels_tf = tf.convert_to_tensor(y_val)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels_tf)).shuffle(1000).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels_tf)).batch(16)

# ---------------------------
# Train Model (15 epochs)
# ---------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Save model
model.save_pretrained('case_legalbert_model')
tokenizer.save_pretrained('case_legalbert_model')

# ---------------------------
# Predict on New Sentences
# ---------------------------
new_sentences = ["I want to breakup with my husband"]

cleaned_sentences = [clean_text(s) for s in new_sentences]
new_inputs = tokenizer(cleaned_sentences, return_tensors='tf', padding=True, truncation=True, max_length=256)

logits = model(new_inputs)[0]
probs = tf.nn.softmax(logits, axis=1).numpy()

print("Predictions for new sentences:")
for sentence, prob in zip(cleaned_sentences, probs):
    top3_idx = np.argsort(prob)[-3:][::-1]
    print("----------------------------------------")
    print(sentence)
    print("----------------------------------------")
    for idx in top3_idx:
        label = label_encoder.inverse_transform([idx])[0]
        confidence = round(100 * prob[idx], 2)
        print(f"{label} - {confidence}%")
