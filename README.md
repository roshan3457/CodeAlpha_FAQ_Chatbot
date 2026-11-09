# CodeAlpha Task 2 — FAQ Chatbot

A small FAQ chatbot built with Python, Streamlit, and scikit-learn. It represents each FAQ question with TF-IDF vectors and uses cosine similarity to select the closest match.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. The FAQ questions are converted into TF-IDF vectors.
2. A user's question is transformed using the same vectorizer.
3. Cosine similarity measures how close the user's question is to each FAQ.
4. The bot returns the answer for the closest match if its score passes a small threshold.

## Customize
Edit the `FAQS` list in `app.py`. Add pairs in this format:
```python
("Your question?", "Your answer.")
```

## Test
Try: `What is machine learning?`, `Which programming language is used?`, and `How do I run this chatbot?`

## Limitations
This is a retrieval-based FAQ chatbot, not a generative LLM. It can only answer questions represented by its FAQ data and may misunderstand unrelated or differently phrased questions.
