#Ch14_9
import pandas as pd
from pandas import Series, DataFrame
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", "Suguru", "Mitsuo"]}

attri_data_frame1 = DataFrame(attri_data1)
attri_data2 = {"ID": ["107", "109"],
               "city": ["Sendai", "Nagoya"],
               "birth_year": [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)


attri_data_frame1.append(attri_data_frame2).sort_values(by="ID", ascending=True).reset_index(drop=True)