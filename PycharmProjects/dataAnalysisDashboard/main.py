import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import requests

# Title
st.title("COVID-19 Global Data Dashboard")
st.sidebar.image("coolcolorswithoutline.png", use_container_width=True)
st.sidebar.write("Explore the impact of COVID-19 globally.")


st.markdown("""
<style>
    .main {
        background-color: #f0f5f9;
    }
    h1 {
        color: #333333;
    }
    h2 {
        color: #555555;
    }
</style>
""", unsafe_allow_html=True)

# reading data from csv and selecting the columns that we need to keep
data = pd.read_csv("country_wise_latest.csv")
columns_to_keep = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']
data = data[columns_to_keep]
data.rename(columns={'Country/Region': 'Country'}, inplace=True)

# creating ui with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Analysis", "Prediction", "Interactive Maps", "Advanced Analysis", "Feedback"])

# Tab 1: Data Analysis
with tab1:
    st.header("COVID-19 Data Analysis")
    countries = data['Country'].unique()
    selected_country = st.selectbox("Select a Country:", countries)

    filtered_data = data[data['Country'] == selected_country]
    st.write(f"Data for {selected_country}:", filtered_data)

    st.subheader(f"COVID-19 Statistics for {selected_country}")
    fig, ax = plt.subplots()
    categories = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    values = filtered_data[categories].values.flatten()
    ax.bar(categories, values, color=plt.cm.Set3(np.arange(len(categories))))
    ax.set_title(f"COVID-19 Cases in {selected_country}")
    ax.set_ylabel("Number of Cases")
    st.pyplot(fig)

# Tab 2: Prediction
with tab2:
    st.header("Prediction Module")
    days = st.slider("Number of Days for Prediction:", 1, 60, 30)
    data['Days'] = np.arange(len(data))

    X = np.array(data['Days']).reshape(-1, 1)
    y = data['Confirmed']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    future_days = np.array(range(len(data), len(data) + days)).reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    predicted_cases = model.predict(future_days_poly)

    fig, ax = plt.subplots()
    ax.plot(range(len(data)), y, label="Actual Cases")
    ax.plot(range(len(data), len(data) + days), predicted_cases, label="Predicted Cases", linestyle="--", color="orange")
    ax.set_title(f"{days}-Day Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Confirmed Cases")
    ax.legend()
    st.pyplot(fig)

# Tab 3: Dynamic interactive maps
with tab3:
    st.header("Interactive Global Maps")

    countries = data["Country"].unique().tolist()
    countries.insert(0, "All Countries")

    selected_countries = st.multiselect(
        "Select Countries to Display",
        options=countries,
        default="All Countries"
    )

    # filtering data
    if "All Countries" in selected_countries:
        filtered_data = data
    else:
        filtered_data = data[data["Country"].isin(selected_countries)]

    # Choropleth Map
    fig = px.choropleth(
        filtered_data,
        locations="Country",
        locationmode="country names",
        color="Confirmed",
        hover_name="Country",
        hover_data={"Confirmed": True},
        title="Global Confirmed Cases",
        color_continuous_scale="Viridis",
    )

    # Map updates
    fig.update_layout(
        title_font=dict(size=24, family='Arial', color='darkgreen'),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="gray",
            projection_type="equirectangular",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )


    fig.update_geos(
        showcountries=True,
        countrycolor="black",
        fitbounds="locations"
    )

    # optimize the zoom of the map and show
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Advanced Analysis
with tab4:
    st.header("Advanced Analysis Tools")
    correlation = data[['Confirmed', 'Deaths', 'Recovered', 'Active']].corr()
    st.write("Correlation Matrix")
    st.dataframe(correlation)

    fig = px.pie(data, names='Country', values='Confirmed', title='Case Distribution by Country')
    st.plotly_chart(fig)

# Tab 5: Feedback
with tab5:
    st.header("We value your feedback!")
    feedback = st.text_area("Your Feedback")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

# Download Data
st.sidebar.header("Download Data")
file_format = st.sidebar.selectbox("Select File Format", ["CSV", "Excel", "JSON"])
if file_format == "CSV":
    st.sidebar.download_button(
        label="Download as CSV",
        data=data.to_csv(index=False),
        file_name="covid_data.csv",
        mime="text/csv",
    )
elif file_format == "Excel":
    st.sidebar.download_button(
        label="Download as Excel",
        data=data.to_excel(index=False, engine='openpyxl'),
        file_name="covid_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
elif file_format == "JSON":
    st.sidebar.download_button(
        label="Download as JSON",
        data=data.to_json(),
        file_name="covid_data.json",
        mime="application/json",
    )


st.sidebar.header("Real-Time Updates")
if st.sidebar.button("Fetch Latest Data"):
    response = requests.get("https://api.covid19api.com/summary")
    if response.status_code == 200:
        covid_data = response.json()
        st.sidebar.success("Data fetched successfully!")
        st.write("Global Summary:", covid_data["Global"])
    else:
        st.sidebar.error("Failed to fetch data.")
