import os
import sys
import time
import pandas as pd
import plotly.express as px 
import streamlit as st
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

csv_folder = "csv_files"  
video_name = None

st.set_page_config(
    page_title="Vehicle Data Dashboard",
    page_icon="🚗",
    layout="wide",
)

class CSVHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.df = None
        self.fig = None
        self.make_model_fig = None
        self.table = None

    def on_created(self, event):
        if event.is_directory:
            return
        filename = os.path.basename(event.src_path)
        if filename == f"{video_name}.csv":
            print("New CSV file created:", event.src_path)
            self.update_dashboard(event.src_path)

    def update_dashboard(self, csv_file_path):
        if os.path.exists(csv_file_path):
            self.df = pd.read_csv(csv_file_path)
            if self.fig is None:
                self.create_elements()
            else:
                self.update_elements()
        else:
            print("CSV file does not exist:", csv_file_path)

    def create_elements(self):
        self.fig = create_histogram(self.df)
        self.make_model_fig = create_make_class_histogram(self.df)
        self.create_dashboard_elements()

    def create_dashboard_elements(self):
        col = st.columns((1, 4, 2.5), gap='medium')
        with col[0]:
            st.markdown(":white[**Total Vehicles**]")
            self.total_vehicles_element = st.empty()
            self.pie_chart_element = st.empty()

        with col[1]:
            self.histogram_element = st.empty()
            self.make_model_histogram_element = st.empty()  # Nouvel élément pour afficher create_make_class_histogram
            self.display_histogram()  # Nouvelle méthode pour afficher les graphiques dans la colonne 1

        with col[2]:
            self.table_element = st.empty()
            self.display_table()

    def update_elements(self):
        self.fig = create_histogram(self.df)
        self.make_model_fig = create_make_class_histogram(self.df)
        self.update_dashboard_elements()

    def update_dashboard_elements(self):
        self.total_vehicles_element.markdown(f"{self.df.shape[0]}")
        self.update_pie_chart()
        self.histogram_element.plotly_chart(self.fig, use_container_width=True)
        self.make_model_histogram_element.plotly_chart(self.make_model_fig, use_container_width=True)
        self.display_table()

    def update_pie_chart(self):
        spain_vehicles = self.df[self.df['Country'] == 'Espagne'].shape[0]
        france_vehicles = self.df[self.df['Country'] == 'France'].shape[0]
        custom_colors = ['#bc6fe3', '#ddf75c']
        fig_pie = px.pie(names=['Espagne', 'France'], 
             values=[spain_vehicles, france_vehicles], 
             title='Spain/France',
             color_discrete_sequence=custom_colors)
        fig_pie.update_layout(title_font_color='white')
        fig_pie.update_traces(hole=0.6, textinfo='percent+label', insidetextorientation='radial')
        self.pie_chart_element.plotly_chart(fig_pie, use_container_width=True)

    def display_histogram(self):
        self.histogram_element.plotly_chart(self.fig, use_container_width=True)
        self.make_model_histogram_element.plotly_chart(self.make_model_fig, use_container_width=True)

    def display_table(self):
        df_display = self.df[['Class', 'Registration', 'Country']]
        self.table_element.table(df_display)


def create_histogram(df):
    all_classes = ['car', 'truck', 'bus', 'motorbike']  
    vehicle_counts = pd.DataFrame({'Class': [], 'Color': []})  # Create an empty DataFrame
    for detected_class in all_classes:
        class_counts = df[df['Class'] == detected_class]['Color'].value_counts().reset_index()
        class_counts.columns = ['Color', 'Count']
        class_counts['Class'] = detected_class
        vehicle_counts = pd.concat([vehicle_counts, class_counts])  # Concatenate data from each class
    
    # Define your own colors for each class
    custom_colors = ['#aee2e8', '#dfbaf5', '#f3fc9f', '#9be89f']

    fig = px.bar(vehicle_counts, x='Class', y='Count', color='Color', color_discrete_sequence=custom_colors)
    fig.update_layout(
        title={
            'text': "Number of Vehicles by Class and Color",
            'y':1, 
            'x':0.5, 
            'xanchor': 'center', 
            'yanchor': 'top' 
        }
    )
    fig.update_yaxes(tick0=0, dtick=1)  # Set y-axis scale without comma
    fig.update_traces(marker_line_width=0)  # Make bars thinner
    fig.update_layout(height=300)  # Adjust figure height
    return fig


def create_make_class_histogram(df):
    # Agréger les données par marque et classe
    make_class_counts = df.groupby(['Make', 'Class']).size().reset_index(name='Count')
    
    # Define your own colors for each class
    custom_colors = ['#aee2e8', '#dfbaf5', '#f3fc9f', '#9be89f']
    
    fig = px.bar(make_class_counts, y='Make', x='Count', color='Class', orientation='h', color_discrete_sequence=custom_colors)

    fig.update_layout(
        title={
            'text': "Number of Vehicles by Make and Class",
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Number of Vehicles", 
        yaxis_title="Make",
    )
    fig.update_traces(marker_line_width=0)  
    fig.update_layout(height=300) 
    return fig


def main():
    global video_name

    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    else:
        st.write("No video name passed as argument.")
        return

    event_handler = CSVHandler()
    observer = Observer()
    observer.schedule(event_handler, path=csv_folder, recursive=False)
    observer.start()

    while True:
        csv_file_path = os.path.join(csv_folder, f"{video_name}.csv")
        event_handler.update_dashboard(csv_file_path)
        time.sleep(10)

if __name__ == "__main__":
    main()
