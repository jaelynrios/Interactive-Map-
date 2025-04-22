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
        background-color: #0e5669;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #000000;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.2em;
        padding: 12px;
        font-weight: bold;
    }

    div.stButton > button {
        background-color: #FFFFFF;
        color: #FFFFFF;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
    }

    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 600;
    }

    .deck-tooltip {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 📊 Load dataset
data = pd.read_csv('san_jose_eih_sites.csv')
data = data.dropna(subset=['latitude', 'longitude'])

# ➕ Calculate Infrastructure Influence Score (IIS)
def calculate_iis(row, w_sent, w_lib, w_hosp):
    norm_sent = row['sentiment_score'] / 100
    norm_lib = 1 - min(row['proximity_to_library'], 1000) / 1000
    norm_hosp = 1 - min(row['proximity_to_hospital'], 1000) / 1000
    return (w_sent * norm_sent) + (w_lib * norm_lib) + (w_hosp * norm_hosp)

# 🧭 Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Home", "🗺️ Map Viewer", "🧠 AI Site Analysis", "📊 Community Pulse", "🧩 Resident Matching", "📈 Post-Site Feedback"])

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
    - 🧩 Match individuals to appropriate sites
    - 📈 Track performance feedback post-deployment
    """)

    file_path = 'san_jose_eih_sites.csv'
    if os.path.exists(file_path):
        modified_time = os.path.getmtime(file_path)
        st.info(f"📅 Dataset last updated: {datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')}")

    with st.expander("📂 Preview Raw Data Table"):
        st.dataframe(data)

# 🗺️ MAP VIEWER TAB
with tab2:
    st.header("📍 Interactive Map of Candidate EIH Sites")

    weight_sentiment = 0.33
    weight_library = 0.33
    weight_hospital = 0.34
    data['iis_score'] = data.apply(lambda row: calculate_iis(row, weight_sentiment, weight_library, weight_hospital), axis=1)

    def tag_site(score):
        if score >= 0.75:
            return "🏅 Ideal"
        elif score >= 0.5:
            return "👍 Moderate"
        else:
            return "⚠️ Poor"

    data['suitability_tag'] = data['iis_score'].apply(tag_site)

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

    tooltip_text = """📍 Site: {site_name}
🏫 Library: {proximity_to_library}m
🏥 Hospital: {proximity_to_hospital}m
💬 Sentiment Score: {sentiment_score}
🔎 Suitability: {suitability_tag}"""

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": tooltip_text}
    )

    st.pydeck_chart(r)

# 🧠 AI ANALYSIS TAB
with tab3:
    st.header("🔍 AI-Powered Site Analysis")
    st.markdown("Select one or more candidate sites below for analysis:")

    selected = st.multiselect("📌 Choose sites to analyze:", options=data['site_name'].unique())

    with st.expander("📘 About this AI Analysis"):
        st.markdown("""
        - The recommendations provided here are generated using **OpenAI's GPT model**.
        - They are based on proximity metrics and sentiment scores in the uploaded dataset.
        - Please note these insights are **not absolute facts**—they are generated based on patterns in the data and may reflect inherent biases.
        - Use them as **a guide**, not a final decision-making tool.
        """)

    st.markdown("⚙️ **Customize Weighting** (optional)")
    weight_sentiment = st.slider("Weight for Sentiment Score", 0.0, 1.0, 0.33)
    weight_library = st.slider("Weight for Proximity to Library", 0.0, 1.0, 0.33)
    weight_hospital = st.slider("Weight for Proximity to Hospital", 0.0, 1.0, 0.34)

    if st.button("🚀 Run AI Analysis") and selected:
        selected_data = data[data['site_name'].isin(selected)]
        selected_data['iis_score'] = selected_data.apply(lambda row: calculate_iis(row, weight_sentiment, weight_library, weight_hospital), axis=1)

        site_summary = ""
        for _, row in selected_data.iterrows():
            site_summary += (
                f"- {row['site_name']}: Library {row['proximity_to_library']}m, "
                f"Hospital {row['proximity_to_hospital']}m, "
                f"Sentiment {row['sentiment_score']}, IIS Score: {row['iis_score']:.2f}\n"
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

Be specific in your reasoning based on the numbers given."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful urban planning assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        st.subheader("📈 AI Recommendation")
        st.success(response.choices[0].message.content)

# 📊 COMMUNITY PULSE TAB
with tab4:
    st.header("📊 Community Sentiment & Infrastructure Pulse")

    st.markdown("This view provides a high-level overview of sentiment and access to resources across the city.")

    st.subheader("Top 3 Sites by Sentiment")
    st.dataframe(data.sort_values(by="sentiment_score", ascending=False).head(3))

    st.subheader("Bottom 3 Sites by Sentiment")
    st.dataframe(data.sort_values(by="sentiment_score", ascending=True).head(3))

    st.subheader("Average Infrastructure Influence Score (IIS)")
    st.metric("Citywide Avg IIS", f"{data['iis_score'].mean():.2f}")

# 🧩 RESIDENT MATCHING TAB
with tab5:
    st.header("🧩 Resident-to-Site Matching")

    st.markdown("""
    This AI-powered tool helps caseworkers or planners assign individuals to the most suitable EIH site based on personal needs.
    """)

    name = st.text_input("Resident Name")
    needs = st.text_area("Describe the resident's needs, preferences, or constraints:")

    if st.button("🧠 Match to Site") and name and needs:
        match_prompt = f"""You are a social housing advisor. Based on the following resident profile and available EIH site data, recommend the best site for placement and explain your reasoning:

Resident Info:
{name}
{needs}

Site Options:
{data[['site_name', 'proximity_to_library', 'proximity_to_hospital', 'sentiment_score']].to_string(index=False)}

Be thoughtful, empathetic, and specific."""

        match_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You match residents with appropriate temporary housing based on their needs."},
                {"role": "user", "content": match_prompt}
            ]
        )

        st.subheader("🔗 Best Match Recommendation")
        st.info(match_response.choices[0].message.content)

# 📈 FEEDBACK LOOP TAB
with tab6:
    st.header("📈 Post-Site Feedback Tracker")

    st.markdown("Use this section to log performance data or feedback about EIH sites post-deployment for future learning.")

    feedback_site = st.selectbox("Select Site for Feedback:", data['site_name'].unique())
    feedback = st.text_area("Enter qualitative or performance-based feedback:")

    if st.button("💾 Save Feedback"):
        with open("site_feedback_log.txt", "a") as f:
            f.write(f"\n{datetime.datetime.now()} | Site: {feedback_site} | Feedback: {feedback}")
        st.success("✅ Feedback logged. Thank you!")

