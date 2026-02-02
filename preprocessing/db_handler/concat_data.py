import sqlite3
from tqdm import tqdm
from itertools import product
conn = sqlite3.connect("../../data/database.db")

cursor = conn.cursor()
tables = ['air_pollution', 'alcohol_consumption', 'BMI', 'cardiovascular_diseases', 'cholesterol', 'diabetes', 'glucose', 'infrastructure', 'physical_activities', 'tobacco']

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

    main_query = f'select id from {exclude} where SpatialDim = ? AND TimeDim = ?'
    main_ids = cursor.execute(main_query, (spatial_dim, time_dim,))

    num_points = 0
    # Ta chỉ lấy ids của điểm dữ liệu để ghép
    output_ids = []
    # Tiến hành điền mẫu sẵn Output_ids
    for row in main_ids:
        output_ids.append(list())
        output_ids[num_points].append((row[0], exclude))
        num_points += 1

    for support in supports:
        query = f'select id from {support} where SpatialDim = ? AND TimeDim = ?'
        characters = cursor.execute(query, (spatial_dim, time_dim,))

        last_id = -1

        index = 0
        for raw_id in characters:
            _id = raw_id[0]
            last_id = _id

            if index >= len(output_ids):
                break

            output_ids[index].append((_id, support))
            index += 1
        
        if index < len(output_ids):
            for _ in range(len(output_ids) - index):
                output_ids[index].append((last_id, support))
                index += 1
    
    # print(f"Số sample ứng riêng cho {spatial_dim} và {time_dim} là: ", len(output_ids))
    return output_ids

total_samples = []
for country, _time in tqdm(product(countries, times), desc="Đang tạo mẫu", total=len(countries) * len(times)):
    total_samples += generate_sample(country, _time)

print("Tổng số sample: ", len(total_samples))
conn.close()

# Tiến hành viết chương trình truy xuất ra số
