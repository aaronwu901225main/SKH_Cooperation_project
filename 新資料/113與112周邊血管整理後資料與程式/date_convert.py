import pandas as pd

def convert_date_format(date):
    """將不同格式的日期轉換為標準 YYYY/MM/DD 格式"""
    if pd.isna(date):
        return None
    date = str(date).strip()
    
    if "/" in date:  # 格式：YYYY/MM/DD
        try:
            return pd.to_datetime(date, format="%Y/%m/%d").strftime("%Y/%m/%d")
        except:
            return date  # 解析失敗則保持原樣
        
    elif len(date) == 8 and date.isdigit():  # 格式：YYYYMMDD
        try:
            return pd.to_datetime(date, format="%Y%m%d").strftime("%Y/%m/%d")
        except:
            return date
        
    elif len(date) == 7 and date.isdigit():  # 格式：YYYMMDD（民國年）
        try:
            year = int(date[:3]) + 1911  # 民國年轉西元年
            return pd.to_datetime(f"{year}{date[3:]}", format="%Y%m%d").strftime("%Y/%m/%d")
        except:
            return date
    
    return date  # 無法識別則保持原樣

# 範例：讀取 Excel 並應用轉換
file_path = "周邊血管手術申報表-113年度.xlsx"  # 修改為你的檔案路徑
sheet_name = "11210_11310"  # 修改為你的工作表名稱
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 確保欄位名稱正確（這裡假設日期欄位名稱為「手術日期」，請根據你的資料修改）
date_column = "手術\n日期"
df[date_column] = df[date_column].apply(convert_date_format)

# 儲存轉換後的 Excel
df.to_excel("converted_dates.xlsx", index=False)
