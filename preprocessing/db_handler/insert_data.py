import sqlite3
import os
import json
import pandas as pd
from tqdm import tqdm
from sqlite3 import Connection

conn = sqlite3.connect("../../data/database.db")

# Tiến hành insert dữ liệu vào database dựa theo trường thời gian Date (
# ràng buộc cập nhật (thứ tự thêm vào là thứ tự xuất hiện)
#)
def insert_data_to_db(table_name, filename, connection : Connection):
    print("Chuẩn bị đọc:", filename)
    df = pd.read_json(filename)
    print("Hoàn thành đọc:", filename)

    # Tiến hành một số bước xử lý
    # Sắp xếp lại thời gian xuất hiện của dữ liệu
    print("Đang xử lý thời gian...")
    df["TimeDimensionBegin"] = pd.to_datetime(df["TimeDimensionBegin"], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
    df["TimeDimensionEnd"] = pd.to_datetime(df["TimeDimensionEnd"], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
    df["NumericValue"] = pd.to_numeric(df["NumericValue"])
    df["Date"] = pd.to_datetime(df["Date"], format='mixed')
    print("Hoàn thành xử lý thời gian...")

    # Tiến hành sắp xếp dữ liệu theo Date
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    # Tiến hành loại bỏ trường Date khỏi thao tác
    df_main = df.drop(columns=["Date"])
    
    query = f'''
    INSERT INTO {table_name} (
        ParentLocationCode, SpatialDim, Value, NumericValue, 
        TimeDimensionBegin, TimeDimensionEnd, TimeDimensionValue, 
        TimeDimType, TimeDim, IndicatorCode
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    print("Đang xử lý...")
    cursor = connection.cursor()
    count = 1
    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="Progress"):
        parent = row["ParentLocationCode"]
        spatial_dim = row["SpatialDim"]
        value = row["Value"]
        numeric_value = row["NumericValue"]
        time_dimension_begin = row["TimeDimensionBegin"]
        time_dimension_end = row["TimeDimensionEnd"]
        time_dimension_value = row["TimeDimensionValue"]
        time_dim_type = row["TimeDimType"]
        time_dim = row["TimeDim"]
        indicator_code = row["IndicatorCode"]

        try:
            cursor.execute(query, (
                parent, spatial_dim, value, numeric_value,
                time_dimension_begin, time_dimension_end,
                time_dimension_value, time_dim_type,
                time_dim, indicator_code
            ))

        except Exception as e:
            print(e)

        if count % 1000 == 0:
            connection.commit()

        count += 1

    connection.commit()
    cursor.close()

    print("Hoàn thành xử lý...")

# Thử nghiệm thao tác
TARGET_FOLDER = "../../data/raw_concat/"
for filename in os.listdir(TARGET_FOLDER):
    tablename = filename.split('.')[0]
    insert_data_to_db(tablename, TARGET_FOLDER + filename, conn)
conn.commit()
conn.close()