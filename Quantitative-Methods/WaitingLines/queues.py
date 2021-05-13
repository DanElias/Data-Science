import math
import pandas as pd
from datetime import time
import datetime
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

class QueueSimulation:
    def __init__(self):
        self.start_time = time(hour = 9, minute = 0)
        self.n = input("Give the number of clients: ")
        self.n = int(self.n)
        if self.n <= 0:
            print("\nThe number of clients has to be greater or equal to 1")
            exit()
        self.columns = [
            'Random Interval Time',
            'Hour of arrival',
            'Random Transaction Time',
            'Service begins at',
            'Service ends at',
            'Client waiting time',
            'ATM idle time'
        ]
        self.total_transaction_time = 0
        self.total_client_waiting_time = 0
        self.total_atm_idle_time = 0
        self.total_waiting_clients = 0
        self.total_usage_time = 0
        index = pd.Series(range(1,self.n + 1))
        self.df = pd.DataFrame(index=index, columns=self.columns)
        self.interarrival_time()
        self.transaction_time()
        self.calculations_arrival()
        self.calculations_service()
        self.add_results_row()
        self.print_final_results()
        

    def print_final_results(self):
        print(self.df)
        print(f'\nAverage waiting time per client: {self.total_client_waiting_time/self.n} minutes to be attended')
        print(f'\nProbability that a client will wait in the line: {100 * (self.total_waiting_clients/self.n) }%', )
        print(f'\nPercentage of the ATM\'s idle time: {100 * (self.total_atm_idle_time/self.total_usage_time)}%')
        print(f'\nAverage waiting time per client: {self.total_transaction_time/self.n} min for transaction')

    def add_results_row(self):
        self.total_transaction_time = self.df['Random Transaction Time'].sum()
        self.total_client_waiting_time = self.df['Client waiting time'].sum()
        self.total_atm_idle_time = self.df['ATM idle time'].sum()
        self.df.loc[self.n + 1] = ['Totals'] + ['.'] + [self.total_transaction_time] + ['.'] + ['.'] + [self.total_client_waiting_time] + [self.total_atm_idle_time]

    def interarrival_time(self):
        rit = [0]
        for i in range(self.n - 1):
            rit.append(random.randint(0,10))
            #rit.append(np.random.uniform())
        self.df["Random Interval Time"] = rit
        
    def transaction_time(self):
        rtt = []
        for i in range(self.n):
            rtt.append(random.randint(1,10))
        self.df["Random Transaction Time"] = rtt
        
    def calculations_arrival(self):
        last_time = self.start_time
        hoa = []
        for index, row in self.df.iterrows():
            if index == 1:
                row["Hour of arrival"] = last_time
            else:
                time_change = datetime.timedelta(minutes=row["Random Interval Time"])
                row["Hour of arrival"] = (
                    datetime.datetime.combine(datetime.date.today(), last_time) + time_change).time()
                last_time = row["Hour of arrival"]
            hoa.append(row["Hour of arrival"])
        self.df["Hour of arrival"] = hoa

    def calculations_service(self):
        last_time = self.start_time
        ss = []
        se = []
        cwt = []
        ait = [] 
        for index, row in self.df.iterrows():

            waiting_time = datetime.datetime.combine(datetime.date.today(), last_time) - datetime.datetime.combine(datetime.date.today(), row["Hour of arrival"])
            
            duration_in_s = waiting_time.total_seconds() 
            
            minutes_difference = int(divmod(duration_in_s, 60)[0])
            if minutes_difference < 0:
                ait.append(abs(minutes_difference))
                cwt.append(0)
            else:
                if minutes_difference != 0:
                    self.total_waiting_clients += 1
                cwt.append(minutes_difference)
                ait.append(0)

            row["Service begins at"] = max(
                (datetime.datetime.combine(datetime.date.today(), last_time)).time(),
                row["Hour of arrival"])

            time_change = datetime.timedelta(minutes=row["Random Transaction Time"])

            row["Service ends at"] = (
                datetime.datetime.combine(datetime.date.today(), row["Service begins at"]) + time_change).time()

            last_time = row["Service ends at"]
            
            ss.append(row["Service begins at"])
            
            se.append(row["Service ends at"])
        self.df["Service begins at"] = ss
        self.df["Service ends at"] = se
        self.df["Client waiting time"] = cwt
        self.df["ATM idle time"] = ait

        usage_time = datetime.datetime.combine(datetime.date.today(), last_time) - datetime.datetime.combine(datetime.date.today(), self.start_time)
        duration_in_s = usage_time.total_seconds() 
        minutes_difference = int(divmod(duration_in_s, 60)[0])
        self.total_usage_time = minutes_difference
            

queue = QueueSimulation()

        