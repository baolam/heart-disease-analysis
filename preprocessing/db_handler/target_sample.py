import sqlite3
import pandas as pd
import datetime
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

def get_common_value(field, excludes=['infrastructure']):
    print("Xử lý trường:", field)
    temps = []

    for table in tables:
        if table in excludes:
            continue
        query = 'select distinct ' + field + ' from ' + table
        tempo = cursor.execute(query)

        _result = set()
        for row in tempo:
            _result.add(row[0])
            temps.append(_result)

        print(table, ':', len(_result), '->')

    output : set = temps[0]
    for temp in temps[1:]:
        output = output.intersection(temp)
    
    print("Hoàn thành trường:", field)
    return output
    
countries = list(get_common_value('SpatialDim'))
print("Tổng số quốc gia chung là:", len(countries))

# Tiến hành viết chương trình truy xuất ra số
managed_samples = sqlite3.connect("../../data/sample_strategy/sample_v2.db")

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

managed_samples.execute('DROP TABLE IF EXISTS NearsestSample_FD')
managed_samples.execute(illustrate_table)
sample_cursor = managed_samples.cursor()

def _get_year(time):
    return int(time.split(' ')[0].split('-')[0])

def generate_sample(spatial_dim, exclude = 'cardiovascular_diseases'):
    supports = tables.copy()
    supports.remove(exclude)

    main_query = f'select id, NumericValue, TimeDimensionBegin from {exclude} where SpatialDim = ? order by id'
    targets = cursor.execute(main_query, (spatial_dim,))

    # Ta chỉ lấy ids của điểm dữ liệu để ghép
    result = []
    # Tiến hành điền mẫu sẵn result
    for row in targets:
        result += [(row[0], row[1], _get_year(row[2]), exclude)]
    
    print("Tổng mẫu ứng với ", spatial_dim, " là:", len(result))

    # Tiến hành cài đặt phương pháp ghép mẫu
    # Ứng với mỗi mẫu target, ta kiếm các feature phù hợp nhất để đẩy vào target đó

    # QUản lí
    ids, data, belongs, times = list(), list(), list(), list()
    counters = defaultdict(int)
    for target in tqdm(result, desc=f"Tạo mẫu cho {spatial_dim}", total=len(result)):

        _ids, _features, _belongs = list(), list(), list()
        for support in supports:
            # Nhãn để tính offset
            label = f'{target[2]}-{support}'
            counters[label] = counters.get(label, -1) + 1

            query1 = f'select id, NumericValue from {support} where SpatialDim = ? AND strftime("%Y", TimeDimensionBegin) = ? ORDER BY id LIMIT 1 OFFSET ?'
            support_feature = cursor.execute(query1, (spatial_dim, str(target[2]), counters[label], ))

            flag = False
            # Lấy chính xác năm để ghép dữ liệu
            for row in support_feature:
                _ids.append(row[0])
                _features.append(row[1])
                _belongs.append(support)
                flag = True
                break
            
            # Không có thì tiến hành điền bằng các năm gần nhất nó
            # Hoặc dùng LinearRegression để hồi quy ra năm ))
            if not flag:
                query2 = f'select id, NumericValue from {support} where SpatialDim = ? AND strftime("%Y", TimeDimensionBegin) BETWEEN ? AND ? ORDER BY id LIMIT 1 OFFSET ?'
                support_feature = cursor.execute(query2, (spatial_dim, str(target[2] - 2), str(target[2] + 2), counters[label], ))

                for row in support_feature:
                    _ids.append(row[0])
                    _features.append(row[1])
                    _belongs.append(support)
                    break
        
        _ids.append(target[0])
        _features.append(target[1])
        _belongs.append(target[3])

        ids.append(_ids)
        data.append(_features)
        belongs.append(_belongs)
        times.append(target[2])
    
    return ids, data, belongs, times

total_samples = 0
output_ids = list()

for spatial in countries:
    ids, data, belongs, times = generate_sample(spatial)
    output_ids += ids

    total_samples += len(data)

    # Định nghĩa một query insert dữ liệu
    query = '''INSERT INTO NearsestSample(
        y, x1, x2, x3, x4, x5, x6, x7, x8, x9,
        SpatialDim, TimeDim
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    
    for num, field, time in zip(data, belongs, times):
        # Tạo định dạng dict để cho lưu ánh xạ dễ dàng hơn (riêng cho từng nhãn cột chỉ định
        # ở trên)
        lookups = defaultdict(float)
        
        # Tiến hành điền dữ liệu theo các trường để tiến hành insert dữ liệu
        for _num, _field_name in zip(num, field):
            lookups[mapping_labels[_field_name]] = _num

        try:
            sample_cursor.execute(query, (
                    lookups.get('y', None), 
                    lookups.get('x1', None), 
                    lookups.get('x2', None), 
                    lookups.get('x3', None), 
                    lookups.get('x4', None), 
                    lookups.get('x5', None), 
                    lookups.get('x6', None), 
                    lookups.get('x7', None), 
                    lookups.get('x8', None), 
                    lookups.get('x9', None), 
                    spatial, time)
                )
        except Exception as e:
            print("Có lỗi phát sinh:", e)

    # output_ids.append(ids)
    # Kết thúc mỗi lần lọc cần ghi dữ liệu
    managed_samples.commit()

cursor.close()
sample_cursor.close()
managed_samples.close()
conn.close()

id_frame = pd.DataFrame(output_ids)
id_frame.to_csv('../../data/sample_strategy/nearest_sample.csv')