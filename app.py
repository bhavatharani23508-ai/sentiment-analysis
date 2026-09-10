import streamlit as st
from transformers import pipeline


# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)


st.title("😊 AI Sentiment Analyzer")
st.write("Enter a sentence and let AI identify its sentiment.")


# Load 3-class sentiment model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


classifier = load_model()


# Analyze sentiment
def analyze_sentiment(text):

    result = classifier(text)[0]

    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        sentiment = "Positive"

    elif label == "negative":
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment, confidence, label


# Input box
text = st.text_area(
    "Enter your text:",
    placeholder="Example: The movie was amazing and I really enjoyed it."
)


# Analyze button
if st.button("🔍 Analyze Sentiment"):

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        with st.spinner("AI is analyzing..."):

            sentiment, confidence, raw_label = analyze_sentiment(text)


        st.subheader("Result")


        if sentiment == "Positive":

            st.success(
                f"😊 Sentiment: {sentiment}"
            )

        elif sentiment == "Negative":

            st.error(
                f"😞 Sentiment: {sentiment}"
            )

        else:

            st.info(
                f"😐 Sentiment: {sentiment}"
            )


        st.write(
            f"Confidence: {confidence:.2%}"
        )


        # Show model output for checking
        with st.expander("🔎 Technical Details"):

            st.write("Model label:", raw_label)
            st.write("Confidence:", f"{confidence:.2%}")