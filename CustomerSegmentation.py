import streamlit as st
import mysql.connector
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import altair as alt
from KModesCluster import perform_clustering

# Open the database connection
def open_database():
    connection = mysql.connector.connect(
        host='mysql-bau-batmanyigitcan-d115.h.aivencloud.com',
        port = '25976',# e.g., 'localhost' or 'your-remote-host'
        user='avnadmin',  # your MySQL username
        password='AVNS_xHYRWCE6hmnxqJG3PA6',  # your MySQL password
        database='defaultdb'  # your MySQL database name
    )
    return connection

# Function to fetch data
def fetch_data(connection):
    query = "SELECT * FROM customer_data"
    data = pd.read_sql(query, connection)
    return data

# Close the database connection
def close_database(connection):
    connection.close()

# Function to display graphs
def display_graphs(data, num_clusters):
    st.subheader("Dataset Preview")
    data_preview = st.empty()
    with st.spinner("Loading data"):
        time.sleep(1)
        data_preview.write(data.head())

    st.sidebar.header("Feature Selection")
    selected_feature = st.sidebar.selectbox("Select a feature to visualize", data.columns[:-1])

    if selected_feature:
        st.subheader(f" Different Cluster Graphs For {selected_feature}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=selected_feature, hue='Cluster', data=data, ax=ax)
        plt.xlabel(selected_feature)
        plt.ylabel('Percentages')
        plt.title(f'Cluster Distribution of {selected_feature}')
        plt.legend(title='Cluster')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Pie chart for categorical
        if data[selected_feature].dtype == 'object':
            for cluster_label in range(num_clusters):
                subset = data[data['Cluster'] == cluster_label]
                pie_fig, pie_ax = plt.subplots(figsize=(6, 6))
                subset[selected_feature].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=pie_ax)
                plt.title(f'Distribution of {selected_feature} for Cluster {cluster_label}')
                st.pyplot(pie_fig)

        # Box plot for numerical
        else:
            box_fig, box_ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=selected_feature, hue='Cluster', data=data, ax=box_ax, palette='Set2', width=.35,whis=(0, 100))
            plt.xlabel('Cluster')
            plt.ylabel(selected_feature)
            plt.title(f'Boxplot of {selected_feature} by Cluster')
            st.pyplot(box_fig)

        # Bar graphs for x-axis product_category
    if data[selected_feature].dtype == 'object':
        st.subheader(f"Visualize Features Relative to {selected_feature}")
        for cluster_label in range(num_clusters):
            cluster_data = data[data['Cluster'] == cluster_label]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(x='product_category', hue=selected_feature, data=cluster_data, ax=ax, palette='Set2')
            plt.title(f"Cluster {cluster_label} - {selected_feature} Distribution")
            plt.xlabel("Product Category")
            plt.ylabel("Percentages")
            plt.xticks(rotation=45)
            plt.legend(title=selected_feature)
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

        # Categorical or Numerical ?
        st.subheader(f"Interactive graphs of {selected_feature}")
        if data[selected_feature].dtype == 'object':  # Categorical feature
            chart = alt.Chart(data).mark_bar(cornerRadius=2).encode(
                x=alt.X(selected_feature, axis=alt.Axis(labelAngle=45)),
                y='count()',
                color='Cluster:N',
                tooltip=[selected_feature, 'Cluster', alt.Tooltip('count()', title='Count')]
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            data[selected_feature] = pd.to_numeric(data[selected_feature], errors='coerce')
            chart = alt.Chart(data).mark_bar(cornerRadius=2).encode(
                x=alt.X(selected_feature, type='quantitative', bin=alt.Bin(maxbins=20), axis=alt.Axis(labelAngle=45)),
                y='count()',
                color='Cluster:N',
                tooltip=[
                    alt.Tooltip(selected_feature, type='quantitative', bin=alt.Bin(maxbins=20), title=selected_feature),
                    'Cluster', alt.Tooltip('count()', title='Count')]
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)

def main():
    st.title("Customer Segmentation")

    connection = open_database()
    data = fetch_data(connection)
    st.sidebar.header("Cluster Slidebar")
    num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=5, value=3)
    clustered_data = perform_clustering(data, num_clusters)
    display_graphs(clustered_data, num_clusters)
    close_database(connection)

if __name__ == "__main__":
    main()