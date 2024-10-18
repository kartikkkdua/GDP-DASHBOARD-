import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages

# Set up page configuration
st.set_page_config(page_title='GDP Dashboard', layout='wide')

# Function to fetch GDP data from the CSV file
def get_gdp_data():
    DATA_FILENAME = Path(__file__).parent / 'gdp_data.csv'
    try:
        raw_gdp_df = pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'gdp_data.csv' is in the same directory.")
        return pd.DataFrame()

    MIN_YEAR = 1980
    MAX_YEAR = 2022
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])
    return gdp_df

# Load GDP data
with st.spinner('Loading GDP data...'):
    gdp_df = get_gdp_data()

# Title and description
st.title('GDP Dashboard')
st.markdown("Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website.")
st.markdown(" A project by Kartik[https://github.com/kartikkkdua]")
st.markdown(" Contact at kartikcodespaces@gmail.com for more info ")

if not gdp_df.empty:
    min_value = gdp_df['Year'].min()
    max_value = gdp_df['Year'].max()

    from_year, to_year = st.slider(
        'Select the range of years:',
        min_value=min_value,
        max_value=max_value,
        value=[min_value, max_value]
    )

    countries = gdp_df['Country Code'].unique()
    selected_countries = st.multiselect(
        'Select countries to display data for:',
        countries,
        ['IND', 'USA', 'JPN']
    )

    # GDP threshold filter
    gdp_threshold = st.number_input('Filter by GDP threshold:', value=0, min_value=0)

    if selected_countries:
        filtered_gdp_df = gdp_df[
            (gdp_df['Country Code'].isin(selected_countries)) &
            (gdp_df['Year'] >= from_year) &
            (gdp_df['Year'] <= to_year) &
            (gdp_df['GDP'] >= gdp_threshold)
        ]

        # Calculate GDP Growth Rate
        filtered_gdp_df['GDP Growth Rate (%)'] = filtered_gdp_df.groupby('Country Code')['GDP'].pct_change() * 100

        # GDP vs Time Graph
        st.header('GDP vs Time Graph')
        plot_type = st.selectbox("Choose Plot Type", ['Line Chart', 'Bar Chart'])
        pivot_gdp = filtered_gdp_df.pivot(index='Year', columns='Country Code', values='GDP')
        if plot_type == 'Line Chart':
            st.line_chart(pivot_gdp)
        elif plot_type == 'Bar Chart':
            st.bar_chart(pivot_gdp)

        # Downloadable graph as PNG
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        for country in selected_countries:
            country_data = filtered_gdp_df[filtered_gdp_df['Country Code'] == country]
            plt.plot(country_data['Year'], country_data['GDP'], label=country)
        plt.xlabel('Year')
        plt.ylabel('GDP')
        plt.title('GDP vs Year')
        plt.legend()
        plt.grid()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.download_button(
            label="Download Graph as PNG",
            data=buf,
            file_name='gdp_graph.png',
            mime='image/png'
        )

        # Save as PDF option
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            plt.figure(figsize=(10, 5))
            for country in selected_countries:
                country_data = filtered_gdp_df[filtered_gdp_df['Country Code'] == country]
                plt.plot(country_data['Year'], country_data['GDP'], label=country)
            plt.xlabel('Year')
            plt.ylabel('GDP')
            plt.title('GDP vs Year')
            plt.legend()
            plt.grid()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

        pdf_buffer.seek(0)
        st.download_button(
            label="Download Graph as PDF",
            data=pdf_buffer,
            file_name='gdp_graph.pdf',
            mime='application/pdf'
        )

        # Country Data Table
        st.subheader('Country Data Table')
        st.dataframe(filtered_gdp_df[['Country Code', 'Year', 'GDP', 'GDP Growth Rate (%)']])

        # Time Series Forecasting
        st.header('GDP Forecasting')
        forecast_years = st.slider('Select number of years to forecast:', min_value=1, max_value=10, value=5)

        # Forecasting Model
        forecast_results = {}
        for country in selected_countries:
            country_data = filtered_gdp_df[filtered_gdp_df['Country Code'] == country]
            if not country_data.empty:
                model = ExponentialSmoothing(country_data['GDP'], trend='add', seasonal='add', seasonal_periods=4)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_years)
                forecast_results[country] = forecast

        # Display Forecast Results
        for country, forecast in forecast_results.items():
            st.subheader(f'GDP Forecast for {country}')
            st.line_chart(forecast)

        # Additional Metrics
        st.subheader('Additional Metrics')
        avg_gdp = filtered_gdp_df['GDP'].mean()
        min_gdp = filtered_gdp_df['GDP'].min()
        max_gdp = filtered_gdp_df['GDP'].max()

        st.metric(label="Average GDP", value=f"${avg_gdp:,.2f}")
        st.metric(label="Minimum GDP", value=f"${min_gdp:,.2f}")
        st.metric(label="Maximum GDP", value=f"${max_gdp:,.2f}")

        # Convert DataFrame to CSV for download
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(filtered_gdp_df)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_gdp_data.csv",
            mime='text/csv'
        )

        # GDP Ranking by Selected Year
        st.header('GDP Ranking by Selected Year')
        selected_year = st.selectbox('Select a Year for GDP Ranking:', range(from_year, to_year + 1))
        yearly_gdp_df = filtered_gdp_df[filtered_gdp_df['Year'] == selected_year]
        ranking_df = yearly_gdp_df.sort_values(by='GDP', ascending=False).reset_index(drop=True)
        ranking_df.index += 1  # Start index at 1

        st.dataframe(
            ranking_df[['Country Code', 'GDP']].style.format({'GDP': '${:,.2f}'})
        )

        # Top N Countries by GDP
        st.header('Top N Countries by GDP')
        top_n = st.slider('Select the number of top countries to display:', 1, len(countries), 10)
        total_gdp_df = filtered_gdp_df.groupby('Country Code')['GDP'].sum().reset_index()
        top_countries_df = total_gdp_df.nlargest(top_n, 'GDP')
        st.bar_chart(top_countries_df.set_index('Country Code')['GDP'])

        # GDP Distribution using Plotly
        st.header('GDP Distribution')
        hist_bins = st.slider('Select the number of bins for the histogram:', min_value=5, max_value=50, value=20)
        fig = px.histogram(
            filtered_gdp_df,
            x='GDP',
            nbins=hist_bins,
            title='GDP Distribution',
            labels={'GDP': 'GDP (in USD)'},
            template='plotly_dark'  # Optional: for a dark-themed plot
        )
        st.plotly_chart(fig)

        # Country Information Pop-up
        st.header('Country Information')
        country_info = {
            'IND': "India is a South Asian country and the seventh-largest by land area.",
            'USA': "The United States is a North American country with the world's largest economy.",
            'JPN': "Japan is an island nation in East Asia known for its advanced technology.",
            # Add more country descriptions as needed
        }
        for country in selected_countries:
            with st.expander(f"More about {country}"):
                st.write(country_info.get(country, "Country information not available."))

        # Compare GDP Between Two Countries
        st.header('Compare GDP Between Two Countries')
        compare_countries = st.multiselect(
            'Select two countries for comparison:',
            countries,
            default=['USA', 'IND']
        )
        if len(compare_countries) == 2:
            compare_df = filtered_gdp_df[filtered_gdp_df['Country Code'].isin(compare_countries)]
            pivot_compare = compare_df.pivot(index='Year', columns='Country Code', values='GDP')
            st.line_chart(pivot_compare)
        else:
            st.warning("Please select exactly two countries for comparison.")

# Footer or additional information
st.markdown("---")
st.markdown("### Additional Resources")
st.markdown("For further analysis, visit the [World Bank Open Data](https://data.worldbank.org/).")
