import sqlite3
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
def generate_sample(spatial_dim, time_dim):
    query = 'select * from cardiovascular_diseases where SpatialDim = ? AND TimeDim = ?'
    result = cursor.execute(query, (spatial_dim, time_dim,))

    for row in result:
        print(row)

generate_sample('VNM', 2010)
conn.close()