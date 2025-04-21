import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import openai
import datetime
from dotenv import load_dotenv

# 🌍 Load environment variables
load_dotenv()

# 🎨 Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-color:rgb(150, 219, 253);
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #0B3D91;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.2em;
        padding: 12px;
        font-weight: bold;
    }

    div.stButton > button {
        background-color: #0B3D91;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
    }

    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 600;
    }

    .deck-tooltip {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 📊 Load dataset
data = pd.read_csv('san_jose_eih_sites.csv')
data = data.dropna(subset=['latitude', 'longitude'])

# 🧭 Create tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🗺️ Map Viewer", "🧠 AI Site Analysis"])

# 🏠 HOME TAB
with tab1:
    st.image("HomeFinderLogo.png", width=200)
    st.markdown("""
    <div style="text-align: center;">
        <img src="HomeFinder Logo.png" width="200">
    </div>
    """, unsafe_allow_html=True)

    st.title("🏙️ San Jose EIH Site Explorer")
    st.markdown("Welcome to the **Emergency Interim Housing (EIH)** Site Explorer powered by AI.")
    st.markdown("""
    This tool allows you to:
    - 🗺️ View candidate EIH sites on a map  
    - 📊 Analyze infrastructure & community sentiment  
    - 🧠 Use AI to recommend optimal sites for development
    """)

    # Feature 2: Data Freshness Indicator
    file_path = 'san_jose_eih_sites.csv'
    if os.path.exists(file_path):
        modified_time = os.path.getmtime(file_path)
        st.info(f"📅 Dataset last updated: {datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')}")

    with st.expander("📂 Preview Raw Data Table"):
        st.dataframe(data)

# 🗺️ MAP VIEWER TAB
with tab2:
    st.header("📍 Interactive Map of Candidate EIH Sites")

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=data,
        get_position='[longitude, latitude]',
        get_radius=200,
        get_color='[200, 30, 0, 160]',
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=data['latitude'].mean(),
        longitude=data['longitude'].mean(),
        zoom=11,
        pitch=0
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "text": "📍 Site: {site_name}\n🏫 Library: {proximity_to_library}m\n🏥 Hospital: {proximity_to_hospital}m\n💬 Sentiment Score: {sentiment_score}"
        }
    )

    st.pydeck_chart(r)

# 🧠 AI ANALYSIS TAB
with tab3:
    st.header("🔍 AI-Powered Site Analysis")
    st.markdown("Select one or more candidate sites below for analysis:")

    selected = st.multiselect("📌 Choose sites to analyze:", options=data['site_name'].unique())

    # Feature 1: AI Transparency Disclaimer
    with st.expander("📘 About this AI Analysis"):
        st.markdown("""
        - The recommendations provided here are generated using **OpenAI's GPT model**.
        - They are based on proximity metrics and sentiment scores in the uploaded dataset.
        - Please note these insights are **not absolute facts**—they are generated based on patterns in the data and may reflect inherent biases.
        - Use them as **a guide**, not a final decision-making tool.
        """)

    # Feature 3: Manual Weighting for AI Evaluation
    st.markdown("⚙️ **Customize Weighting** (optional)")
    weight_sentiment = st.slider("Weight for Sentiment Score", 0.0, 1.0, 0.33)
    weight_library = st.slider("Weight for Proximity to Library", 0.0, 1.0, 0.33)
    weight_hospital = st.slider("Weight for Proximity to Hospital", 0.0, 1.0, 0.34)

    if st.button("🚀 Run AI Analysis") and selected:
        selected_data = data[data['site_name'].isin(selected)]

        site_summary = ""
        for _, row in selected_data.iterrows():
            site_summary += (
                f"- {row['site_name']}: Library {row['proximity_to_library']}m, "
                f"Hospital {row['proximity_to_hospital']}m, "
                f"Sentiment {row['sentiment_score']}\n"
            )

        weight_summary = f"""
User-defined weights:
- Sentiment: {weight_sentiment}
- Library Proximity: {weight_library}
- Hospital Proximity: {weight_hospital}
"""

        prompt = f"""You are a policy analyst. Analyze the following Emergency Interim Housing (EIH) candidate sites based on proximity to infrastructure and resident sentiment. Recommend which sites seem more viable and why:

{site_summary}

User-defined priority:
{weight_summary}

Be specific in your reasoning based on the numbers given.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful urban planning assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        st.subheader("📈 AI Recommendation")
        st.success(response.choices[0].message.content)
