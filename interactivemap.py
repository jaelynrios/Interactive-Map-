import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import openai
from dotenv import load_dotenv

# 🌍 Load environment variables
load_dotenv()

# 🎨 Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #CBF3F9;
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
    # 🏠 HOME TAB HEADER

    # 🖼️ Display Logo – Streamlit native (for accessibility)
    st.image("HomeFinderLogo.png", width=200)

    # 🎨 Centered HTML version (for visual control)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="HomeFinder Logo.png" width="200">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("🏙️ San Jose EIH Site Explorer")
    st.markdown("Welcome to the **Emergency Interim Housing (EIH)** Site Explorer powered by AI.")
    st.markdown("""
    This tool allows you to:
    - 🗺️ View candidate EIH sites on a map  
    - 📊 Analyze infrastructure & community sentiment  
    - 🧠 Use AI to recommend optimal sites for development
    """)

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

    if st.button("🚀 Run AI Analysis") and selected:
        selected_data = data[data['site_name'].isin(selected)]

        site_summary = ""
        for _, row in selected_data.iterrows():
            site_summary += (
                f"- {row['site_name']}: Library {row['proximity_to_library']}m, "
                f"Hospital {row['proximity_to_hospital']}m, "
                f"Sentiment {row['sentiment_score']}\n"
            )

        prompt = f"""You are a policy analyst. Analyze the following Emergency Interim Housing (EIH) candidate sites based on proximity to infrastructure and resident sentiment. Recommend which sites seem more viable and why:

{site_summary}

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
