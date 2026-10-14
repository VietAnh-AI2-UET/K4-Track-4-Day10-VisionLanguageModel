Họ tên: **Bùi Việt Anh**
`dataset_version`: `d10-21e4070301`

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | | | | | | | | |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

- Số lượng tham số của Qwen3-VL-4B:
    - ViT: ~306 triệu.
    - Projector: ~109 triệu, gồm merger ~27 triệu và các DeepStack merger ~82 triệu.
    - LLM: ~4,0 tỷ. 
    - Tổng mô hình ~4,44 tỷ.

- E1 cập nhật LLM
- E2 cập nhật LLM + Projector
- E3 cập nhật LLM + Projector + ViT

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
<điền>

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
<điền>

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

- Mục tiêu là học cách trả lời đúng JSON chứa lớp và tọa độ, nên chỉ tính loss trên phần đáp án.
- Code gán `labels = -100` cho phần yêu cầu và token ảnh để bỏ qua chúng khi tính loss.
- Nếu tính cả những phần này, model phải học dự đoán lại đầu vào, làm phân tán việc học đáp án và có thể giảm chất lượng.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

- Projector là khối chuyển thông tin ảnh sang dạng model ngôn ngữ sử dụng được. Trọng số nén 4-bit không hỗ trợ cập nhật toàn bộ trực tiếp trong cách huấn luyện đang dùng.
- Code dùng `llm_int8_skip_modules=["visual", "lm_head"]` để giữ khối thị giác, gồm projector, ngoài phần nén 4-bit.
- Ở E2/E3, trọng số projector được chuyển sang `float32` để cập nhật ổn định và phù hợp với `GradScaler` (công cụ điều chỉnh độ lớn khi tính cập nhật). Các phép tính vẫn có thể dùng `float16` để tiết kiệm bộ nhớ.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

- Các ảnh cùng phiên quay thường rất giống nhau. Chia ngẫu nhiên có thể đưa ảnh gần trùng vào cả tập học và tập kiểm tra, khiến F1 cao nhưng không phản ánh khả năng xử lý cảnh mới.
- Chia theo phiên quay giúp mỗi phiên chỉ thuộc một tập.
- Theo `data/splits/meta.json`, phiên bản `d10-21e4070301`, số ảnh và số box là:
- **Train:** 1.422 ảnh; ổ gà **1.074**, vết nứt **1.750**, nắp cống **691**.
- **Validation:** 276 ảnh; ổ gà **66**, vết nứt **470**, nắp cống **103**.
- **Test:** 311 ảnh; ổ gà **121**, vết nứt **299**, nắp cống **163**.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
