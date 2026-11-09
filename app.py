import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CodeAlpha FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")
st.write("Ask a question about the sample internship/project information. The bot finds the closest matching FAQ using TF-IDF and cosine similarity.")

FAQS = [
    ("What is CodeAlpha?", "CodeAlpha is an organization that offers learning and internship opportunities. Check its official communication for current program details."),
    ("What is an internship?", "An internship is a short-term learning experience where students gain practical skills by working on projects."),
    ("Which programming language is used?", "This chatbot is developed using Python."),
    ("What is artificial intelligence?", "Artificial intelligence is a field of computing focused on systems that perform tasks commonly associated with human intelligence."),
    ("What is machine learning?", "Machine learning is a branch of AI in which models learn patterns from data to make predictions or decisions."),
    ("What is NLP?", "Natural Language Processing (NLP) helps computers process, understand, and generate human language."),
    ("How do I run this chatbot?", "Install the packages with pip install -r requirements.txt, then run streamlit run app.py."),
    ("Can I add my own questions?", "Yes. Add more question-and-answer pairs to the FAQS list in app.py."),
    ("What is Python?", "Python is a popular, high-level programming language known for readable syntax and a large ecosystem of libraries."),
    ("How can I contact support?", "Use the official contact details provided by your organization or internship coordinator.")
]

questions = [item[0] for item in FAQS]
answers = [item[1] for item in FAQS]
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
faq_matrix = vectorizer.fit_transform(questions)

def get_answer(user_question):
    query_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(query_vector, faq_matrix)[0]
    best_index = scores.argmax()
    confidence = float(scores[best_index])
    if confidence < 0.12:
        return "Sorry, I don't know the answer to that yet. Try asking about Python, AI, machine learning, NLP, or the project.", confidence, None
    return answers[best_index], confidence, questions[best_index]

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("Ask me a question...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    answer, confidence, matched = get_answer(question)
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()

if st.session_state.history:
    last_user = next((m["content"] for m in reversed(st.session_state.history) if m["role"] == "user"), None)
    if last_user:
        answer, confidence, matched = get_answer(last_user)
        with st.expander("Show matching details"):
            st.write(f"Similarity score: {confidence:.3f}")
            if matched:
                st.write(f"Matched FAQ: {matched}")

with st.sidebar:
    st.subheader("Example questions")
    for q in questions:
        st.write("• " + q)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
