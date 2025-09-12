# E2E-Person-Alert-FaceRec

Hệ thống E2E: Phát hiện người → Cắt mặt → Nhận diện → Cảnh báo thời gian thực (webcam).

## Cấu trúc
env/, detector/, collector/, recognizer/, app/, data/, mlflow/

## Cách chạy nhanh
```bash
pip install -r env/requirements.txt
python recognizer/train.py
python recognizer/export_onnx.py
python app/realtime_alert.py
