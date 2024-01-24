import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import emoji  # Import the emoji library
import plotly.graph_objects as go  # Import Plotly
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import pandas as pd  # Import pandas for working with DataFrames
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import gdown

custom_palette = sns.color_palette("Set3")

# Custom CSS for WhatsApp Chat Analyzer App (Dark Theme)
custom_css = """
<style>
/* Body Styles */
body {
    background-color: #121212; /* Dark Gray */
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #ffffff; /* White */
}

/* Sidebar Styles */
.sidebar .sidebar-content {
    background-color: #1e1e1e; /* Darker Gray */
    color: #ffffff; /* White */
    padding: 1rem; 
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Sidebar Header Styles */
.sidebar .sidebar-content .stMarkdown {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 1.5rem;
}

/* Widget Label Styles */
.Widget>label {
    color: #ffffff; /* White */
}

/* Expander Header Styles */
.streamlit-expanderHeader {
    background-color: #424242; /* Gray */
    color: #ffffff; /* White */
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
}

/* Expander Content Styles */
.streamlit-expanderContent {
    background-color: #2b2b2b; /* Darker Gray */
    color: #ffffff; /* White */
    padding: 0.5rem 1rem;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Button Styles */
.stButton>button {
    background-color: #3a33aa;
    color: #ffffff; /* White */
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
}

/* Selectbox Styles */
.stSelectbox>div:first-child {
    background-color: #2b2b2b; /* Darker Gray */
    color: #ffffff; /* White */
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
}

/* Data Table Styles */
.stDataFrame, .stTable {
    background-color: #2b2b2b; /* Darker Gray */
    color: #ffffff; /* White */
}

/* Plotly Chart Styles */
.stPlotlyChart .plotly .modebar {
    background-color: #2b2b2b; /* Darker Gray */
}

/* Title Styles */
h1 {
    color: #CCCCCC; /* Light gray */
}

/* Subtitle Styles */
h2 {
    color: #FFFFFF; /* White */
}

/* Section Title Styles */
h3 {
    color: #CCCCCC; /* Light gray */
}

/* Section Subtitle Styles */
h4 {
    color: #FFFFFF; /* White */
}

/* Paragraph Styles */
p {
    margin-bottom: 1rem;
}

/* Wordcloud Image Styles */
img {
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
}
</style>
"""


# Set page title
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Set page title
# st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

# Sidebar
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

sns.set_palette(custom_palette)
plt.style.use("ggplot")


if uploaded_file is not None:
    # Read and preprocess data
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", num_messages)
        with col2:
            st.metric("Total Words", words)
        with col3:
            st.metric("Media Shared", num_media_messages)
        with col4:
            st.metric("Links Shared", num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        plt.xlabel("Month")
        plt.ylabel("Number of Messages")
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        plt.xlabel("Date")
        plt.ylabel("Number of Messages")
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            plt.xlabel("Day of Week")
            plt.ylabel("Number of Messages")
            st.pyplot(fig)

        with col2:
            st.subheader("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            plt.xlabel("Month")
            plt.ylabel("Number of Messages")
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Day of the Week")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                plt.xlabel("User")
                plt.ylabel("Number of Messages")
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        plt.axis('off')  # Remove axes
        st.pyplot(fig)

        # Most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        st.title('Most common words')
        st.pyplot(fig)

        # Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            plt.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
            st.pyplot(fig)

        # Load your Keras sentiment analysis model
        # Google Drive link to the model file
        drive_link = 'https://drive.google.com/file/d/1oWNTs-xSyWdJxtdG3NZnNZ0saa5sCc0W/view'

        # Download the model file from Google Drive
        output = 'model.keras'
        gdown.download(drive_link, output, quiet=False)

        # Load the Keras model
        loaded_model = load_model(output)

        max_sequence_length = 30

        # Sentiment analysis for all users
        st.title("Sentiment Analysis Using Keras model")
        sentiment_data = []

        for user in user_list:
            if user == "Overall":
                continue
            user_messages = df[df["user"] == user]["message"].values

            # Tokenize and preprocess the user messages
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(user_messages)
            sequences = tokenizer.texts_to_sequences(user_messages)
            input_data = pad_sequences(sequences, maxlen=max_sequence_length)

            # Perform inference using the Keras model
            sentiment_scores = loaded_model.predict(input_data)
            print(sentiment_scores)
            user_sentiment = sentiment_scores.mean()

            sentiment_text = "Positive" if user_sentiment > 0.5 else "Negative"
            sentiment_data.append((user, sentiment_text))

        sentiment_df = pd.DataFrame(sentiment_data, columns=["User", "Sentiment"])
        st.dataframe(sentiment_df)

        # Overall Sentiment Analysis
        st.title("Overall Sentiment Analysis")
        overall_messages = df["message"].values

        # Tokenize and preprocess the overall messages
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(overall_messages)
        overall_sequences = tokenizer.texts_to_sequences(overall_messages)
        overall_input_data = pad_sequences(overall_sequences, maxlen=max_sequence_length)

        # Perform inference using the Keras model
        overall_sentiment_scores = loaded_model.predict(overall_input_data)
        overall_sentiment = overall_sentiment_scores.mean()

        overall_sentiment_text = "Positive" if overall_sentiment > 0.5 else "Negative"
        st.subheader("Overall Sentiment")
        st.write(f"The overall sentiment of the chat is: {overall_sentiment_text}")


        # Sentiment analysis for all users
        st.title("Sentiment Analysis using inbuilt TextBlob")
        sentiment_data = []

        for user in user_list:
            if user == "Overall":
                continue
            user_messages = df[df["user"] == user]["message"].values
            sentiment_scores = [TextBlob(message).sentiment.polarity for message in user_messages]
            user_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            sentiment_text = "Positive" if user_sentiment >=0 else "Negative"
            sentiment_data.append((user, sentiment_text))

        sentiment_df = pd.DataFrame(sentiment_data, columns=["User", "Sentiment"])
        st.dataframe(sentiment_df)

        # Overall Sentiment Analysis
        st.title("Overall Sentiment Analysis")
        overall_sentiment_scores = [TextBlob(message).sentiment.polarity for message in df["message"].values]
        overall_sentiment = sum(overall_sentiment_scores) / len(overall_sentiment_scores)
        overall_sentiment_text = "Positive" if overall_sentiment >=0 else "Negative"
        st.subheader("Overall Sentiment")
        st.write(f"The overall sentiment of the chat is: {overall_sentiment_text}")
