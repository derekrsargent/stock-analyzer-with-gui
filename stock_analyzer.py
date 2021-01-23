import pandas as pd
import yfinance as yf
from datetime import date
import datetime
from dateutil.relativedelta import *
import numpy as np
from pathlib import Path
import os
import multiprocessing
import threading
import sys
import time
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
import tkinter as tk
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import seaborn as sns
import mplcursors


class GetData:
    def __init__(self):
        pass

    @staticmethod
    def get_all_tickers():
        file_folder = './data/nasdaqlisted.txt'
        file_name = open(file_folder, 'r')
        ticker_list_df = pd.read_csv(file_folder, sep="|", header=0)
        ticker_list = ticker_list_df['Symbol'].tolist()
        file_name.close()
        return ticker_list

    @staticmethod
    def get_filtered_tickers():
        file_folder = './filtered_list.txt'
        file_name = open(file_folder, 'r')
        ticker_list_df = pd.read_csv(file_folder, sep="|", header=None)
        ticker_list_df.columns = ['Tickers', 'Remove']
        ticker_list = ticker_list_df['Tickers'].tolist()
        ticker_list_filtered = []
        file_name.close()

        # Filter ticker list to include only tickers >$3 and <$15
        current = date.today()
        go_back = current - relativedelta(days=3)

        for ticker in ticker_list:
            data = yf.download(ticker.split(',')[0],
                               start=go_back.strftime("%Y-%m-%d"),
                               end=current.strftime("%Y-%m-%d"),
                               threads=True)
            if data.shape[0] > 0:
                if 3 <= data['Adj Close'][-1] <= 15:
                    ticker_list_filtered.append(ticker.split(',')[0])

        return ticker_list_filtered

    def update_filtered_list(self):
        ticker_list_to_filter = self.get_all_tickers()
        with open('limited1.txt', 'w') as writer:
            for i in tqdm(range(len(ticker_list_to_filter))):
                ticker = yf.Ticker(ticker_list_to_filter[i])
                try:
                    if ticker.info['sharesOutstanding'] < 10E6:
                        writer.write(ticker_list_to_filter[i])
                        writer.write(",")
                        writer.write(str(ticker.info['sharesOutstanding']))
                        writer.write("|\n")
                except:
                    pass

        return ticker_list_to_sort

    @staticmethod
    def get_data(ticker):
        days_max = 500
        current = date.today()

        # Since the graph starts 50 weeks in the past but we need 50 weeks of data before that to start
        start = current - relativedelta(days=days_max+250)
        current += relativedelta(days=1)
        sys.stdout = open(os.devnull, "w")
        data = yf.download(ticker,
                           start=start.strftime("%Y-%m-%d"),
                           end=current.strftime("%Y-%m-%d"),
                           threads=True)
        sys.stdout = sys.__stdout__
        data_length = len(data['Volume'])

        # If not enough data return all zeros
        if data_length < 500:
            blank = [0] * data_length
            data['wma50'] = blank
            data['wma20'] = blank
            data['wma10'] = blank
            return data

        # We need to do this because yf.download doesn't return values on weekends/holidays
        data = data.drop(data.index[0:(data_length - days_max)])

        # 250 is 50 weeks, 100 is 20 weeks, 50 is 10 weeks
        counter_start = 0
        counter_end = days_max - 250  # 250

        wma50_data = [0] * (days_max - 250)
        wma20_data = [0] * (days_max - 250)
        wma10_data = [0] * (days_max - 250)

        while counter_end < days_max:
            avg_wma50 = data['Adj Close'][counter_start:counter_end].mean()
            avg_wma20 = data['Adj Close'][(counter_start + 150):counter_end].mean()
            avg_wma10 = data['Adj Close'][(counter_start + 200):counter_end].mean()
            counter_start += 1
            counter_end += 1
            wma50_data.append(avg_wma50)
            wma20_data.append(avg_wma20)
            wma10_data.append(avg_wma10)

        data['wma50'] = wma50_data
        data['wma20'] = wma20_data
        data['wma10'] = wma10_data
        data = data.drop(data.index[0:(days_max - 250)])

        return data


class MainApplication(tk.Frame):
    @staticmethod
    def convert_npdt64_to_dt(t_in):
        # This methods will accept either a single numpy <datetime64> object or a list of them
        t_out = []
        try:
            for t in t_in:
                unix_epoch = np.datetime64(0, 's')
                one_second = np.timedelta64(1, 's')
                seconds_since_epoch = (t - unix_epoch) / one_second
                t_out.append(datetime.datetime.utcfromtimestamp(seconds_since_epoch).strftime("%Y-%m-%d"))
        except TypeError:
            unix_epoch = np.datetime64(0, 's')
            one_second = np.timedelta64(1, 's')
            seconds_since_epoch = (t_in - unix_epoch) / one_second
            t_out.append(datetime.datetime.utcfromtimestamp(seconds_since_epoch).strftime("%Y-%m-%d"))
        return t_out

    def draw(self, ticker):
        # This try/except block will clear the canvas if one already exists
        try:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas_vol.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()
        except AttributeError:
            pass

        sns.set_style("ticks")
        sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 1})
        data = GetData.get_data(ticker)
        fig = Figure(figsize=(7, 3), dpi=100)
        fig_vol = Figure(figsize=(7, 1), dpi=100)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(ticker, fontweight='bold', fontsize='14')
        ax_vol = fig_vol.add_subplot(111)
        ax_vol.set_xlabel('Date')
        ax_vol.set_ylabel('Volume')
        mplcursors.cursor(hover=True)

        xlab = ax.xaxis.get_label()
        ylab = ax.yaxis.get_label()
        xlab.set_style('italic')
        xlab.set_size(10)
        ylab.set_style('italic')
        ylab.set_size(10)
        xlab_vol = ax_vol.xaxis.get_label()
        ylab_vol = ax_vol.yaxis.get_label()
        xlab_vol.set_style('italic')
        xlab_vol.set_size(10)
        ylab_vol.set_style('italic')
        ylab_vol.set_size(10)

        add_converted_dt_to_df = MainApplication.convert_npdt64_to_dt(data.index.values)
        data['Dates'] = add_converted_dt_to_df
        data.plot(kind='line', legend=True, x='Dates', y='Adj Close', ax=ax, label='Closing Price')
        data.plot(kind='line', legend=True, x='Dates', y='wma50', ax=ax, label='50-Week Moving Avg')
        data.plot(kind='line', legend=True, x='Dates', y='wma20', ax=ax, label='20-Week Moving Avg')
        data.plot(kind='line', legend=True, x='Dates', y='wma10', ax=ax, label='10-Week Moving Avg')
        data.plot(kind='line', legend=True, x='Dates', y='Volume', ax=ax_vol, label='Volume')

        self.canvas = FigureCanvasTkAgg(fig, master=self.frm)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas_vol = FigureCanvasTkAgg(fig_vol, master=self.frm)
        self.canvas_vol.draw()
        self.canvas_vol.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent  # tk.Frame sets self.master = parent for you
        self.canvas = None
        self.canvas_vol = None
        self.toolbar = None

        root.title("stonks")
        root.geometry("800x600")
        root.resizable(0, 0)  # Makes it non-resizeable

        listbox = tk.Listbox(root, width=10)
        listbox.place(x=0, y=0)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar = tk.Scrollbar(root)
        scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

        # This frame is for the graph since FigureCanvasTkAgg has no place(x=,y=) attribute
        self.frm = tk.Frame(root, width=1200, height=1000)
        self.frm.pack(side=tk.RIGHT, expand=1)
        self.frm.place(x=80, y=0)

        tickers = GetData().get_filtered_tickers()
        for ticker in tickers:
            listbox.insert(tk.END, ticker.split(',')[0])

        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        listbox.bind('<<ListboxSelect>>', self.on_click)

    def on_click(self, event):
        widget = event.widget
        selection = widget.curselection()
        value = widget.get(selection[0])
        self.draw(value)  # MainApplication.draw(self, value) if not using self


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
