# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.

---

## Cài đặt và chạy pipeline (Bài làm học viên)

### 1. Corpus & cấu trúc bài nộp

- **Corpus thật:** `starter/data/Principles-of-Data-Science-WEB.pdf` — OpenStax *Principles of Data Science* (Ault, Liao, Musolino, 2025, `CC BY-NC-SA 4.0`, `569` trang PDF tổng), trích **107 trang nội dung chính** (`Chapter 1` p.19–61, `Chapter 2` p.67–103, `Chapter 8` p.387–413). Chi tiết manifest: [`data/manifest.json`](data/manifest.json).
- **Mã nguồn pipeline (`starter/src/`):**
  - [`starter/src/ingestion.py`](starter/src/ingestion.py): đọc PDF bằng `pypdf`, chuẩn hóa `NFKC`, xóa header/footer lặp lại, gắn `chapter` / `section` / `book_page` và log cảnh báo.
  - [`starter/src/chunking.py`](starter/src/chunking.py): `fixed_size(500, overlap=50)` (Strategy A) và `by_paragraph(max_size=800)` giữ heading (Strategy B & C).
  - [`starter/src/embeddings.py`](starter/src/embeddings.py): `OpenRouterEmbedder` gọi `openai/text-embedding-3-small` (`1536` chiều, symmetric).
  - [`starter/src/store.py`](starter/src/store.py): `PgVectorStore` lưu vào PostgreSQL `16.16` + `pgvector 0.8.7`, idempotent upsert theo `document_id` + dọn chunk lỗi thời, tìm kiếm cosine `<=>` và hỗ trợ SQL `WHERE metadata->>'chapter' = %s` (Strategy C).
  - [`starter/src/evaluation.py`](starter/src/evaluation.py) & [`starter/src/run.py`](starter/src/run.py): chạy toàn bộ thí nghiệm A/B/C trên benchmark đã khóa và xuất kết quả.
- **Benchmark & báo cáo:**
  - Câu hỏi & vị trí neo mạch lạc (chốt trước thí nghiệm): [`benchmark/questions.json`](benchmark/questions.json) (`15` dev / `5` test, gồm `5` câu metadata filter), [`benchmark/coherence_anchors.json`](benchmark/coherence_anchors.json) (`10` anchors).
  - Nhật ký thí nghiệm & kết quả: [`artifacts/experiment_log.csv`](artifacts/experiment_log.csv), [`results/`](results/) (`metrics_summary.json`, `retrieval_summary.csv`, `traces_dev_test.json`, `coherence_samples.json`, `test_audit_samples.json`, `ingestion_log.json`, `manual_audit.json`, `manual_audit.csv`).
  - Báo cáo cá nhân: [`report/individual/2A202603478_VuTriDung.md`](report/individual/2A202603478_VuTriDung.md).

### 2. Hướng dẫn cài đặt và chạy lại

1. **Khởi động PostgreSQL + pgvector bằng Docker:**

   ```powershell
   cd starter
   docker compose -p day6_student_starter up -d --wait
   ```

2. **Tạo biến môi trường `starter/.env` (không commit file `.env`):**

   ```powershell
   # Windows PowerShell
   Copy-Item .env.example .env
   ```

   ```bash
   # Linux / macOS
   cp .env.example .env
   ```

   Điền `OPENROUTER_API_KEY=<your-key>` vào `starter/.env` (giữ `DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab` và `EMBEDDING_MODEL=openai/text-embedding-3-small`).

3. **Tạo môi trường ảo Python và cài thư viện:**

   *Phiên bản đã kiểm chứng:* Python `3.14.0`, `psycopg==3.3.8`, `pypdf==6.21.4`, `python-dotenv==1.2.4`, `requests==2.34.2`.

   ```powershell
   # Windows PowerShell (trong thư mục starter/)
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe scripts\check_setup.py
   ```

   ```bash
   # Linux / macOS (trong thư mục starter/)
   python3 -m venv .venv
   ./.venv/bin/python -m pip install -r requirements.txt
   ./.venv/bin/python scripts/check_setup.py
   ```

   Kết quả mong đợi:

   ```text
   Database: day6_lab | PostgreSQL: 16.16 (Debian 16.16-1.pgdg12+1) | pgvector: 0.8.7
   Setup OK. Pipeline functions still need to be implemented.
   ```

   *(Lưu ý: thông báo dòng thứ hai là chuỗi tĩnh sẵn có của script starter để xác nhận kết nối Postgres + pgvector.)*

4. **Chạy toàn bộ pipeline ingestion, lưu vector và đánh giá benchmark:**

   ```powershell
   # Windows PowerShell (trong thư mục starter/)
   .\.venv\Scripts\python.exe -m src.run
   ```

   ```bash
   # Linux / macOS (trong thư mục starter/)
   ./.venv/bin/python -m src.run
   ```

   Kết quả chạy thực tế sẽ cập nhật các file trong `results/` và `starter/results/metrics_summary.json`. (Chi phí đo được trong lần chạy thí nghiệm báo cáo là `$0.002264` USD với `openai/text-embedding-3-small` qua OpenRouter.)
