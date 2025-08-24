# LeadLag Project

## Giới thiệu
Bài tiểu luận môn học Máy học (Machine Learning).

## Yêu cầu hệ thống
- Python 3.10 (khuyên dùng **conda** hoặc **venv**)
- pip >= 24.0
- setuptools >= 68.0
- wheel >= 0.41

## Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/trungcvp/leadLag.git
cd <repo>
```

### 2. Cài đặt thư viện

####  Dùng Conda
```bash
conda env create -f environment.yml
conda activate leadlag
```

####  Dùng venv
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# hoặc:
.venv\Scripts\activate         # Windows

# Cài dependencies
pip install -r requirements-pre.txt
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Kiểm tra
python -m pip check
```

---

##  Cấu trúc mã nguồn

###  Code chính (trong `/code/models`)
- **`herm_matrix_algo.py`**: tính ma trận Hermitian  
- **`lead_lag_measure.py`**: bổ sung cách thêm trọng số Volume  
- **`lead_lag_measure_new.py`**: thay đổi cách tính distance correlation  
  *(alpha × cor_returns + (1 - alpha) × cor_volume)*  

###  Các file thực nghiệm
- **`/run_lead_lag/*`**: các file chạy riêng lẻ, tạo ma trận *lead_lag* theo từng phương pháp → lưu thành `*.pkl`  
- **`prepare_data.py`**: chuẩn bị dữ liệu `return` + `volume` → sinh ra 2 file:
  - `returns_matrix.csv`
  - `volume.csv`
- **`experiments.ipynb`**: notebook chạy test kết quả của các mô hình

---

##  Thực nghiệm
- Có thể chạy trực tiếp bằng:  
  ```bash
  jupyter notebook experiments.ipynb
  ```
- Hoặc chạy lại toàn bộ pipeline từ đầu để tái tạo ma trận *Lead-Lag* (sẽ tốn thời gian).  
- Thay thế file ma trận `*.pkl` tương ứng để kiểm tra kết quả của từng lần thực nghiệm.

