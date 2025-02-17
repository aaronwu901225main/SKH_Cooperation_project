import pandas as pd

def filter_excel_rows(input_file, output_file, sheet_name):
    # 讀取 Excel 文件
    xls = pd.ExcelFile(input_file)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 定義 K 到 W 直行的範圍（基於 0 索引）
    columns_K_W = df.columns[10:23]  # K 是第 10 列，W 是第 22 列
    
    # 移除 K~W 範圍內所有值都是 0 的行
    df_filtered = df[~(df[columns_K_W].fillna(0).astype(str) == '0').all(axis=1)]
    
    # 儲存過濾後的數據到新的 Excel 文件
    df_filtered.to_excel(output_file, sheet_name=sheet_name, index=False)
    print(f"已儲存過濾後的數據到 {output_file}")

# 設定檔案路徑和工作表名稱
input_file = "周邊血管手術申報表-113年度.xlsx"
output_file = "過濾後_周邊血管手術申報表.xlsx"
sheet_name = "11210_11310"

# 執行過濾函數
filter_excel_rows(input_file, output_file, sheet_name)
