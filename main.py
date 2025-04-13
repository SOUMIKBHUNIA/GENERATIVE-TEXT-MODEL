# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import random

# Step 1: Prepare the Dataset
# Example dataset (you can replace this with your own dataset)
texts = [
    "Artificial intelligence is a wonderful field that is developing rapidly.",
    "Machine learning models can be used to solve complex problems.",
    "Deep learning is a subset of machine learning that uses neural networks.",
    "Natural language processing allows computers to understand human language.",
    "Recurrent neural networks are great for sequential data like text."
]

# Step 2: Preprocess the Data
# Initialize a tokenizer to convert words into integer tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)  # Fit the tokenizer on the dataset
total_words = len(tokenizer.word_index) + 1  # Adding 1 for out-of-vocabulary words

# Convert texts to sequences of integers (each word becomes an integer)
sequences = tokenizer.texts_to_sequences(texts)

# Create input-output pairs for training
input_sequences = []
for seq in sequences:
    for i in range(1, len(seq)):
        n_gram_sequence = seq[:i+1]  # Create subsequences (e.g., [1, 2], [1, 2, 3], etc.)
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform input length
max_sequence_len = max([len(seq) for seq in input_sequences])  # Find the maximum sequence length
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')  # Pad sequences

# Split into predictors (X) and label (y)
X = input_sequences[:, :-1]  # All but the last word in each sequence
y = input_sequences[:, -1]   # The last word in each sequence

# One-hot encode the labels (convert the target word into a categorical format)
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Step 3: Build the LSTM Model
# Define the model architecture
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),  # Embedding layer to convert words into dense vectors
    LSTM(150, return_sequences=False),  # LSTM layer to process sequential data
    Dense(total_words, activation='softmax')  # Output layer to predict the next word
])

# Compile the model
# Use categorical cross-entropy loss since this is a multi-class classification problem
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the Model
# Train the model on the input-output pairs
model.fit(X, y, epochs=100, verbose=1)  # Train for 100 epochs (you can adjust this)

# Step 5: Generate Text Based on User Prompt
def generate_text(seed_text, next_words=50):
    """
    Generate text based on a seed prompt.
    
    Args:
        seed_text (str): The initial text to start generating from.
        next_words (int): The number of words to generate.
    
    Returns:
        str: The generated text.
    """
    for _ in range(next_words):
        # Convert the seed text into a sequence of integers
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad the sequence to match the input length of the model
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict the probabilities for the next word
        predicted_probs = model.predict(token_list, verbose=0)
        
        # Choose the word with the highest probability
        predicted_index = np.argmax(predicted_probs, axis=1)[0]
        
        # If the model predicts an unknown word (index 0), stop generation
        if predicted_index == 0:
            break
        
        # Convert the predicted index back to a word
        output_word = tokenizer.index_word[predicted_index]
        
        # Append the predicted word to the seed text
        seed_text += " " + output_word
    
    return seed_text

# Step 6: Test the Model with User Prompts
# Provide a user prompt to generate text
user_prompt = "artificial intelligence"
generated_text = generate_text(user_prompt, next_words=20)  # Generate 20 words
print("Generated Text:\n", generated_text)
