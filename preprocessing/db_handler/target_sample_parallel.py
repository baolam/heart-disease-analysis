import sqlite3
import concurrent.futures as futures
from tqdm import tqdm
from collections import defaultdict
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
SOURCE_DB = "../../data/database.db"
TARGET_DB = "../../data/sample_strategy/sample_v3.db"
MAX_WORKERS = 8  # Tùy số nhân CPU của bạn (nên từ 4-8)

# --- THIẾT LẬP NHÃN ---
tables = ['air_pollution', 'alcohol_consumption', 'BMI', 'cardiovascular_diseases', 
          'cholesterol', 'diabetes', 'glucose', 'infrastructure', 'physical_activities', 'tobacco']
labels = ['x1', 'x2', 'x3', 'y', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
mapping_labels = {table: label for table, label in zip(tables, labels)}

def _get_year(time_str):
    try:
        return int(time_str.split(' ')[0].split('-')[0])
    except:
        return 0

# --- HÀM XỬ LÝ TRONG LUỒNG (WORKER) ---
def worker_task(spatial_dim, thread_pos):
    """Mỗi thread tự mở kết nối ĐỌC từ SOURCE_DB"""
    try:
        local_conn = sqlite3.connect(SOURCE_DB)
        local_cursor = local_conn.cursor()
        
        exclude = 'cardiovascular_diseases'
        supports = [t for t in tables if t != exclude]
        
        # 1. Lấy dữ liệu mục tiêu
        main_query = f'SELECT id, NumericValue, TimeDimensionBegin FROM {exclude} WHERE SpatialDim = ? ORDER BY id'
        targets = local_cursor.execute(main_query, (spatial_dim,)).fetchall()

        all_ids, all_rows = [], []
        counters = defaultdict(int)

        # 2. Thanh tiến trình con cho từng quốc gia
        with tqdm(total=len(targets), desc=f" └─ {spatial_dim[:10]}", 
                  position=thread_pos, leave=False, unit="rec") as pbar_inner:
            
            for target in targets:
                t_id, t_val, t_time_raw = target[0], target[1], target[2]
                t_year = _get_year(t_time_raw)
                
                _ids, _features, _belongs = [], [], []
                
                for support in supports:
                    label = f'{t_year}-{support}'
                    offset = counters[label]
                    counters[label] += 1

                    # Tìm đúng năm
                    q1 = f'SELECT id, NumericValue FROM {support} WHERE SpatialDim = ? AND strftime("%Y", TimeDimensionBegin) = ? ORDER BY id LIMIT 1 OFFSET ?'
                    res = local_cursor.execute(q1, (spatial_dim, str(t_year), offset)).fetchone()

                    # Nếu không có, tìm lân cận +/- 2 năm
                    if not res:
                        q2 = f'SELECT id, NumericValue FROM {support} WHERE SpatialDim = ? AND strftime("%Y", TimeDimensionBegin) BETWEEN ? AND ? ORDER BY id LIMIT 1 OFFSET ?'
                        res = local_cursor.execute(q2, (spatial_dim, str(t_year-2), str(t_year+2), offset)).fetchone()

                    if res:
                        _ids.append(res[0])
                        _features.append(res[1])
                        _belongs.append(support)

                # Gộp dữ liệu bảng chính
                _ids.append(t_id)
                _features.append(t_val)
                _belongs.append(exclude)

                # Map dữ liệu vào cột tương ứng (y, x1, x2...)
                lookups = defaultdict(float)
                for val, field_name in zip(_features, _belongs):
                    lookups[mapping_labels[field_name]] = val
                
                row = (
                    lookups.get('y'), lookups.get('x1'), lookups.get('x2'),
                    lookups.get('x3'), lookups.get('x4'), lookups.get('x5'),
                    lookups.get('x6'), lookups.get('x7'), lookups.get('x8'),
                    lookups.get('x9'), spatial_dim, t_year
                )
                
                all_ids.append(_ids)
                all_rows.append(row)
                pbar_inner.update(1)

        local_conn.close()
        return all_ids, all_rows
    except Exception as e:
        return None, str(e)

# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    # 1. Khởi tạo danh sách quốc gia
    print("--- Đang khởi tạo dữ liệu ---")
    conn_init = sqlite3.connect(SOURCE_DB)
    # Lấy giao của SpatialDim từ các bảng (trừ infrastructure)
    all_spatial = []
    for table in [t for t in tables if t != 'infrastructure']:
        res = conn_init.execute(f"SELECT DISTINCT SpatialDim FROM {table}").fetchall()
        all_spatial.append(set(r[0] for r in res))
    countries = sorted(list(set.intersection(*all_spatial)))
    conn_init.close()
    print(f"Tìm thấy {len(countries)} quốc gia chung.")

    # 2. Chuẩn bị Database đích
    if not os.path.exists(os.path.dirname(TARGET_DB)):
        os.makedirs(os.path.dirname(TARGET_DB))
        
    managed_samples = sqlite3.connect(TARGET_DB)
    managed_samples.execute("PRAGMA journal_mode=WAL;")
    managed_samples.execute("PRAGMA synchronous=NORMAL;")
    managed_samples.execute('DROP TABLE IF EXISTS NearsestSample')
    managed_samples.execute('''
        CREATE TABLE NearsestSample(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            y REAL, x1 REAL, x2 REAL, x3 REAL, x4 REAL, 
            x5 REAL, x6 REAL, x7 REAL, x8 REAL, x9 REAL,
            SpatialDim TEXT, TimeDim INTEGER
        )
    ''')

    # 3. Thực thi đa luồng
    insert_query = 'INSERT INTO NearsestSample(y,x1,x2,x3,x4,x5,x6,x7,x8,x9,SpatialDim,TimeDim) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)'
    total_saved = 0

    

    # Thanh tiến trình tổng ở dòng 0
    with tqdm(total=len(countries), desc="TỔNG TIẾN ĐỘ", position=0, unit="quốc gia") as pbar_main:
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Gán thread_pos để tqdm các luồng không đè nhau (dòng 1 trở đi)
            future_to_spatial = {
                executor.submit(worker_task, s, (i % MAX_WORKERS) + 1): s 
                for i, s in enumerate(countries)
            }
            
            for future in futures.as_completed(future_to_spatial):
                spatial = future_to_spatial[future]
                ids, rows = future.result()
                
                if ids is not None:
                    if rows:
                        managed_samples.executemany(insert_query, rows)
                        managed_samples.commit()
                        total_saved += len(rows)
                    pbar_main.set_postfix({"mẫu_đã_ghi": total_saved})
                else:
                    tqdm.write(f"\n[LỖI] Quốc gia {spatial}: {rows}")
                
                pbar_main.update(1)

    managed_samples.close()
    print(f"\n--- HOÀN TẤT ---")
    print(f"Tổng số bản ghi đã lưu: {total_saved}")
    print(f"File lưu tại: {TARGET_DB}")

if __name__ == "__main__":
    main()