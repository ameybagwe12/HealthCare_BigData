import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set options
np.set_printoptions(suppress=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Importing datasets
train = pd.read_csv('./ntrarogyaseva.csv')


# Homepage
def homepage():
    st.title("Healthcare Analytics App")
    st.write(
        "The goal of this project is to accurately predict the Length of Stay for each patient so that the hospitals can optimize resources and function better.")
    st.write("Use the navigation links above to explore the data and perform data analysis.")
    st.markdown("""
        <h2>Amey Bagwe</h2>
        <h2>Abhay Bhosale</h2>
        <h2>Himanshu Bhundere</h2>
        """, unsafe_allow_html=True)


# Data Exploration Section
def data_exploration():
    st.title("Data Exploration")

    # Overview of Data
    st.subheader("Overview of Data")
    st.dataframe(train.head())


    # Fill missing values with the mode
    train.fillna(train.mode().iloc[0], inplace=True)

    # Data visualization
    st.subheader("Data Visualization")

    st.write("Distribution of Sex")
    mappings = {'MALE': 'Male', 'FEMALE': 'Female', 'Male(Child)': 'Boy', 'Female(Child)': 'Girl'}
    train['SEX'] = train['SEX'].replace(mappings)
    st.bar_chart(train['SEX'].value_counts())

    # District-wise analysis
    st.write("District-wise Data Analysis")
    district = st.text_input("Enter the district for the data analysis:")
    if district:
        st.subheader(f"Data Analysis for District: {district}")
        district_data = train[train['DISTRICT_NAME'] == district]
        st.dataframe(district_data.head())

    st.markdown("""
        <h2>Top Disease in Each District</h2>
        """, unsafe_allow_html=True)
    for i in train['DISTRICT_NAME'].unique():
        disease_count = train[train['DISTRICT_NAME'] == i]['SURGERY'].value_counts().head(1)
        most_common_disease = disease_count.index[0]
        count_str = str(disease_count.values[0])  # Convert count to a string
        st.write(
            f"District: {i}\nDisease: {most_common_disease}")
        st.write(f"Count - {count_str}")
        st.markdown("""
        <hr/>
        """, unsafe_allow_html=True)

    
    st.markdown("""
        <h2>Average Claim Amount by District</h2>
        """, unsafe_allow_html=True)
    for i in train['DISTRICT_NAME'].unique():
        st.write(
            f"District: {i}")
        st.write(f"Average Claim Amount: ₹{train[train['DISTRICT_NAME'] == i]['CLAIM_AMOUNT'].mean():.2f}")
        st.markdown("""
        <hr/>
        """, unsafe_allow_html=True)

    # Data analysis by user input
    st.write("Data Analysis by User Input")
    available_variables = ['CATEGORY_NAME', 'SURGERY', 'DISTRICT_NAME', 'HOSP_NAME']
    fetch = st.selectbox("Select a variable to fetch:", available_variables)

    if fetch:
        data_analysis = train.groupby(fetch)[['AGE', 'PREAUTH_AMT', 'CLAIM_AMOUNT']].mean()
        st.dataframe(data_analysis)

    # Specify the format for the date
    date_format = "%d-%m-%Y %H:%M"
    # Convert the 'SURGERY_DATE' column to a datetime format using the specified format
    train['SURGERY_DATE'] = pd.to_datetime(train['SURGERY_DATE'], format=date_format)
    # Extract the year from the date of surgery
    train['Year_of_Surgery'] = train['SURGERY_DATE'].dt.year
    # Streamlit app
    st.title("Count Surgeries by Year")
    # Specify the particular year you want to count surgeries for
    available_years = range(2013, 2018)
    desired_year = st.selectbox("Select a year for the total count of Surgery:", available_years)
    # Count the number of surgeries for the specified year
    total_surgeries_in_desired_year = len(train[train['Year_of_Surgery'] == desired_year])
    st.write(f"Total number of surgeries in {desired_year}: {total_surgeries_in_desired_year}")

# Data Preparation Section
def data_preparation():
    st.title("Data Preparation")

    # Round AGE values
    train_copy = train.copy()
    train_copy['AGE'] = train_copy['AGE'].round(-1)
    st.write("Age Groups and Most Common Surgery")
    for i in sorted(train_copy['AGE'].unique()):
        st.write(f"Age Group: {i}")
        surgery_count = train[train_copy['AGE'] == i]['CATEGORY_NAME'].value_counts().head(1)
        most_common_surgery = surgery_count.index[0]
        count_str = str(surgery_count.values[0])  # Convert count to a string
        st.write(f"Most Common Surgery and Count: {most_common_surgery} - {count_str}")
        st.markdown("""
        <hr/>
        """, unsafe_allow_html=True)

    # Create a Seaborn countplot and save it as an image
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train, x='CATEGORY_NAME', palette='viridis')
    plt.title('Distribution of Stay Categories')
    plt.xlabel('Stay Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig("stay_categories_countplot.png")
    st.image("stay_categories_countplot.png")

    age_visitors = train.groupby('AGE')['CLAIM_AMOUNT'].sum().reset_index()
    st.write("Relationship between Age and Total Visitors")

    # Create a Matplotlib figure and save it as an image
    plt.figure(figsize=(12, 6))
    plt.fill_between(age_visitors['AGE'], age_visitors['CLAIM_AMOUNT'], color='skyblue', alpha=0.7)
    plt.plot(age_visitors['AGE'], age_visitors['CLAIM_AMOUNT'], marker='o', color='blue', label='Visitors')
    plt.xlabel('AGE')
    plt.ylabel('CLAIM AMOUNT')
    plt.title('Relationship between Age and Total Visitors')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.savefig("age_visitors_plot.png")
    st.image("age_visitors_plot.png")

    department_counts = train['CATEGORY_NAME'].value_counts()
    st.write("Distribution of Departments")

    # Create a Matplotlib figure for the pie chart and save it as an image
    plt.figure(figsize=(8, 8))
    plt.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Departments')
    plt.axis('equal')
    plt.savefig("departments_pie_chart.png")
    st.image("departments_pie_chart.png")

    st.write("Distribution of Claim Amount")

    # Create a Matplotlib figure for the histogram and save it as an image
    plt.figure(figsize=(12, 6))
    plt.hist(train['CLAIM_AMOUNT'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Claim Amount')
    plt.xlabel('Claim Amount')
    plt.ylabel('Count')
    plt.savefig("claim_amount_histogram.png")
    st.image("claim_amount_histogram.png")

# Main app
def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Data Exploration", "Data Preparation"]
    page = st.sidebar.selectbox("Go to", pages)

    if page == "Home":
        homepage()
    elif page == "Data Exploration":
        data_exploration()
    elif page == "Data Preparation":
        data_preparation()


if __name__ == "__main__":
    main()
