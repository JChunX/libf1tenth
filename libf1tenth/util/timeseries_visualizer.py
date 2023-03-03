import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimeSeriesVisualizer:
    '''
    TimeseriesVisualizer

    Loads a csv file with timeseries data.
    Timeseries dataformat:
    columns: ["elapsed time", "receive time", "header.stamp", "topic", "value"]
    Plots timeseries data value on a single, interactive plot for all unique topics using header.stamp as the timestamp.
    '''

    def __init__(self, filepath):
        '''
        Initializes TimeSeriesVisualizer class instance with the filepath to the csv file
        
        Parameters:
            filepath (str): Filepath to the csv file containing the timeseries data
        '''
        self.data = pd.read_csv(filepath)
        
    def plot_timeseries(self):
        '''
        Plots timeseries data
        
        Returns:
            matplotlib figure: Interactive figure containing timeseries data
        '''
        # Group data by topic
        topic_data = self.data.groupby('topic')
        
        # Get unique topics and number of topics
        topics = topic_data.groups.keys()
        num_topics = len(topics)
        
        # Set up figure and axes
        fig, ax = plt.subplots()
        
        # Loop through each topic, plot data, and set axis labels and title
        for i, topic in enumerate(topics):
            topic_df = topic_data.get_group(topic)
            ax.plot(topic_df['header.stamp'], topic_df['value'], label=topic)
            
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Data')
        
        plt.show()
        return fig
    
if __name__ == '__main__':
    # Create TimeSeriesVisualizer instance and plot timeseries data
    tsv = TimeSeriesVisualizer('~/Downloads/plot_data.csv')
    tsv.plot_timeseries()
