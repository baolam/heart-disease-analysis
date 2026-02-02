import sqlite3
conn = sqlite3.connect("../../data/database.db")

cursor = conn.cursor()

# Tiến hành tạo ra các bảng dữ liệu
cursor.execute('''
    CREATE TABLE IF NOT EXISTS air_pollution (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS alcohol_consumption(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS BMI(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS cardiovascular_diseases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS cholesterol(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS diabetes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS glucose(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS infrastructure(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS physical_activities(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS tobacco(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ParentLocationCode TEXT,
        SpatialDim TEXT,
        Value TEXT,
        NumericValue REAL,
        TimeDimensionBegin DATETIME,
        TimeDimensionEnd DATETIME,
        TimeDimensionValue TEXT,
        TimeDimType TEXT,
        TimeDim INTEGER,
        IndicatorCode TEXT
    )
''')

conn.commit()
conn.close()