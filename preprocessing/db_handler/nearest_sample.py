import sqlite3
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import defaultdict
conn = sqlite3.connect("../../data/database.db")

cursor = conn.cursor()

tables = ['air_pollution', 'alcohol_consumption', 'BMI', 'cardiovascular_diseases', 'cholesterol', 'diabetes', 'glucose', 'infrastructure', 'physical_activities', 'tobacco']

# Định nghĩa labels ứng riêng cho tables
labels = ['x1', 'x2', 'x3', 'y', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

# Chương trình sinh ra bộ ánh xạ
mapping_labels = dict()
for i, field in enumerate(tables):
    mapping_labels[field] = labels[i]

def get_common_value(field):
    print("Xử lý trường:", field)
    temps = []

    for table in tables:
        query = 'select distinct ' + field + ' from ' + table
        tempo = cursor.execute(query)

        _result = set()
        for row in tempo:
            _result.add(row[0])
            temps.append(_result)

        print(table, ':', len(_result))

    output : set = temps[0]
    for temp in temps[1:]:
        output = output.intersection(temp)
    
    print("Hoàn thành trường:", field)
    return output
    
countries = list(get_common_value('SpatialDim'))
times = list(get_common_value('TimeDim'))

print("Tổng số quốc gia chung là:", len(countries))
print("Tổng số điểm thời gian chung là:", len(times))
print("Các thời gian:", times)

# Lấy các điểm dữ liệu trong đây làm mẫu ghép
objective = tables[3]
def generate_sample(spatial_dim, time_dim, exclude = 'cardiovascular_diseases'):
    supports = tables.copy()
    supports.remove(exclude)

    main_query = f'select id, NumericValue from {exclude} where SpatialDim = ? AND TimeDim = ?'
    main_ids = cursor.execute(main_query, (spatial_dim, time_dim,))

    num_points = 0
    # Ta chỉ lấy ids của điểm dữ liệu để ghép
    result = []
    # Tiến hành điền mẫu sẵn result
    for row in main_ids:
        result.append(list())
        result[num_points].append((row[0], row[1], exclude))
        num_points += 1

    for support in supports:
        query = f'select id, NumericValue from {support} where SpatialDim = ? AND TimeDim = ?'
        characters = cursor.execute(query, (spatial_dim, time_dim,))

        last_id, last_value = -1, -1

        index = 0
        for raw_id in characters:
            _id = raw_id[0]
            last_id = _id
            last_value = raw_id[1]

            if index >= len(result):
                break

            result[index].append((_id, raw_id[1], support))
            index += 1
        
        # Trong trường hợp mà số điểm hỗ trợ ít thì mặc định 
        # lấy điểm cuối
        if index < len(result):
            for _ in range(len(result) - index):
                result[index].append((last_id, last_value, support))
                index += 1
    
    # print(f"Số sample ứng riêng cho {spatial_dim} và {time_dim} là: ", len(result))
    # Định dạng dữ liệu trả về: ứng với từng sample mục tiêu ta có 10 kết quả từ 10 bảng
    # dữ liệu
    return result

# Tiến hành viết chương trình truy xuất ra số
managed_samples = sqlite3.connect("../../data/sample_strategy/samples.db")

illustrate_table = '''
    CREATE TABLE IF NOT EXISTS NearsestSample(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        y REAL,
        x1 REAL,
        x2 REAL,
        x3 REAL,
        x4 REAL,
        x5 REAL,
        x6 REAL,
        x7 REAL,
        x8 REAL,
        x9 REAL,
        SpatialDim TEXT,
        TimeDim INTEGER
    )
'''

managed_samples.execute('DROP TABLE IF EXISTS NearsestSample')
managed_samples.execute(illustrate_table)
sample_cursor = managed_samples.cursor()

total_samples = 0
output_ids = []

for country, _time in tqdm(product(countries, times), desc="Đang tạo mẫu", total=len(countries) * len(times)):
    _samples = generate_sample(country, _time)
    total_samples += len(_samples)

    # Tiến hành lấy riêng và ghép ra dữ liệu
    ids = list()
    # Tạo định dạng dict để cho lưu ánh xạ dễ dàng hơn (riêng cho từng nhãn cột chỉ định
    # ở trên)
    lookups = defaultdict(float)

    # Định nghĩa một query insert dữ liệu
    query = '''INSERT INTO NearsestSample(
        y, x1, x2, x3, x4, x5, x6, x7, x8, x9,
        SpatialDim, TimeDim
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    
    for sample in _samples:
        # Tiến hành điền dữ liệu theo các trường để tiến hành insert dữ liệu
        for _id, _num, _field_name in sample:
            ids.append(_id)
            lookups[mapping_labels[_field_name]] = _num

        try:
            sample_cursor.execute(query, (lookups['y'], lookups['x1'], lookups['x2'], lookups['x3'], lookups['x4'], lookups['x5'], lookups['x6'], lookups['x7'], lookups['x8'], lookups['x9'], country, _time))
        except Exception as e:
            print("Có lỗi phát sinh:", e)

    # output_ids.append(ids)
    # Kết thúc mỗi lần lọc cần ghi dữ liệu
    managed_samples.commit()

print("Tổng số sample: ", total_samples)

# Tiến hành ghi file quản lí về ids
# CSV nhãn đang bị lỗi, cần fix lại!!! )))
# id_frame = pd.DataFrame(output_ids)
# id_frame.to_csv('../../data/sample_strategy/nearest_sample.csv')

conn.close()
managed_samples.close()