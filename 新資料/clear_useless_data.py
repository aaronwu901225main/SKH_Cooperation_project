import pandas as pd

def filter_excel_rows(input_file, output_file, removed_rows_file, sheet_name):
    # 讀取 Excel 文件，確保病歷號碼以字串格式讀取
    xls = pd.ExcelFile(input_file)
    df = pd.read_excel(xls, sheet_name=sheet_name, dtype={"病歷號碼": str})
    
    # 定義 K 到 W 直行的範圍（基於 0 索引）
    columns_K_W = df.columns[10:23]  # K 是第 10 列，W 是第 22 列
    
    # 找出需要刪除的行
    mask = (df[columns_K_W].fillna(0).astype(str) == '0').all(axis=1)
    df_removed = df[mask]
    df_filtered = df[~mask]
    
    # 儲存過濾後的數據到新的 Excel 文件
    df_filtered.to_excel(output_file, sheet_name=sheet_name, index=False)
    
    # 儲存被刪除的行到 CSV 文件
    df_removed.to_csv(removed_rows_file, index=False)
    
    print(f"已儲存過濾後的數據到 {output_file}")
    print(f"已儲存被刪除的數據到 {removed_rows_file}")
    print(f"總共刪除了 {len(df_removed)} 行")

# 設定檔案路徑和工作表名稱
input_file = "周邊血管手術申報表-113年度.xlsx"
output_file = "過濾後_周邊血管手術申報表.xlsx"
removed_rows_file = "刪除的行.csv"
sheet_name = "11210_11310"

# 執行過濾函數
filter_excel_rows(input_file, output_file, removed_rows_file, sheet_name)
