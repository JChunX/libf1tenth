import numpy as np
import pandas as pd


class TimeSeriesReader:
    '''
    Loads a csv file with Foxglove timeseries data.
    Timeseries dataformat:
    columns: ["elapsed time", "receive time", "header.stamp", "topic", "value"]
    
    Plots timeseries data value on a single, interactive plot for all unique topics using header.stamp as the timestamp.
    '''

    def __init__(self, filepath):
        '''
        Initializes TimeSeriesReader class instance with the filepath to the csv file
        
        Parameters:
            filepath (str): Filepath to the csv file containing the timeseries data
        '''
        df = pd.read_csv(filepath)
        # sort df by topic
        self.topic_data = df.groupby('topic')
        self.topics = self.topic_data.groups.keys()
        
    def get_by_topic(self, topic):
        '''
        Retrieves timeseries data for a specific topic
        
        Args:
        - topic (str): Topic to retrieve timeseries data for
        
        Returns:
        - stamp (np.array): Timestamps for the timeseries data
        - values (np.array): Values for the timeseries data
        '''
        topic_df = self.topic_data.get_group(topic)
        values = topic_df['value'].to_numpy()
        stamp = topic_df['header.stamp'].to_numpy()
        
        return stamp, values
        
    def plot_timeseries(self, topics=None):
        import matplotlib.pyplot as plt
        '''
        Plots timeseries data for select topics
        
        Args:
        - topics (list): List of topics to plot timeseries data for
        
        Returns:
        - fig (matplotlib figure): Interactive figure containing timeseries data
        '''
        
        # Set up figure and axes
        fig, ax = plt.subplots()
        
        if topics is None:
            topics = self.topics
        else:
            topics = [topic for topic in topics if topic in self.topics]
            
        # Loop through each topic, plot data, and set axis labels and title
        for _, topic in enumerate(topics):
            topic_df = self.topic_data.get_group(topic)
            ax.plot(topic_df['header.stamp'], topic_df['value'], label=topic)
            
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Data')
        
        plt.show()
        return fig
    
if __name__ == '__main__':
    # Create TimeSeriesReader instance and plot timeseries data
    tsv = TimeSeriesReader('~/Downloads/accel.csv')
    stamp, accel_x = tsv.get_by_topic('accel.x')
    stamp, accel_y = tsv.get_by_topic('accel.y')
