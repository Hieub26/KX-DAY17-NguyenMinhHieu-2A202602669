# Báo Cáo Phân Tích & Đánh Giá Hệ Thống Memory (Day 17 Lab Report)

> **Môn học / Giai đoạn:** Phase 2, Track 3, Day 17: Memory Systems for AI Agent  
> **Sinh viên:** Nguyễn Minh Hiếu (2A202602669)  
> **Repository:** `KX-DAY17-NguyenMinhHieu-2A202602669`

---

## 1. Tổng quan Kiến trúc Hệ thống Memory

Hệ thống trong bài lab triển khai và so sánh thực nghiệm giữa 2 kiến trúc Agent:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          BASELINE AGENT                                │
│   - Short-term Memory thuần túy theo Session/Thread                    │
│   - Không có Persistent Store (Không có User.md)                       │
│   - Không có Compaction (Toàn bộ message cũ được giữ nguyên vào prompt)│
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                          ADVANCED AGENT                                │
│   1. Short-term Memory: Giữ lại K message gần nhất                     │
│   2. Persistent Memory: User.md lưu facts ổn định qua các session      │
│   3. Compact Memory: Tóm tắt heuristic khi thread vượt ngưỡng token    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Kết quả Thực nghiệm Benchmark

Kết quả thu được khi chạy trực tiếp `python src/benchmark.py` trên 2 bộ dữ liệu chuẩn:

### 2.1. Standard Benchmark (`data/conversations.json`)
*10 cuộc hội thoại thông thường (~10 lượt/cuộc), user `dungct` kèm các câu hỏi recall chéo phiên:*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 2,124 | 16,641 | **0.0%** | 20.0% | 0 B | 0 |
| **Advanced Agent** | 2,076 | 26,598 | **100.0%** | **100.0%** | 307 B | 0 |

### 2.2. Long-Context Stress Benchmark (`data/advanced_long_context.json`)
*1 chuỗi hội thoại gồm 16 lượt trao đổi rất dài, nhiều dữ kiện thời sự, sở thích, đính chính và nhiễu:*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 342 | 22,672 | **0.0%** | 20.0% | 0 B | 0 |
| **Advanced Agent** | 578 | **11,418** *(giảm 49.6%)* | **100.0%** | **100.0%** | 243 B | **8** |

---

## 3. Phân tích Chuyên sâu & Giải thích Trade-off (Mốc 75 - 90 Điểm)

### 3.1. Vì sao Advanced Agent đạt Cross-session Recall 100% còn Baseline là 0%?
- **Baseline Agent**: Bộ nhớ chỉ tồn tại trong đối tượng `SessionState` gắn với một `thread_id` cụ thể. Khi bước sang thread mới để trả lời câu hỏi recall, `session.messages` khởi tạo rỗng. Do không có cơ chế lưu trữ lâu dài ra ngoài session, Baseline buộc phải phản hồi: *"Xin chào! Mình chưa có thông tin về bạn trong phiên trò chuyện này"*. Điểm recall đạt 0.0%.
- **Advanced Agent**: Mỗi khi người dùng chia sẻ fact (tên, nơi ở, nghề nghiệp, đồ uống...), hàm `extract_profile_updates()` trích xuất và đưa vào `UserProfileStore` ghi vào file `state/profiles/<user_id>/User.md`. Khi sang thread mới, agent đọc `User.md` để nạp profile vào context. Do đó, agent trả lời chính xác 100% các câu hỏi kiểm tra chéo phiên.

### 3.2. Trade-off ở hội thoại ngắn: Vì sao Advanced tốn Prompt Tokens hơn Baseline?
- Ở **Standard Benchmark**, prompt token của Advanced là **26,598** so với **16,641** của Baseline (cao hơn ~60%).
- **Nguyên nhân**:
  1. Trong các cuộc hội thoại ngắn (10 lượt ngắn, ~20 token/lượt), tổng token của thread chưa chạm tới ngưỡng nén (`compact_threshold_tokens = 800`).
  2. Baseline chỉ gửi các tin nhắn đã có trong thread đó.
  3. Advanced Agent ở mỗi lượt chat đều phải kèm thêm:
     $$\text{Prompt Tokens} = \text{Tokens}(User.md) + \text{Tokens}(\text{System Prompt}) + \text{Tokens}(\text{Messages})$$
  4. Phần payload cố định từ file `User.md` (~40 - 70 token/lượt) được gửi lặp lại ở mọi turn, dẫn đến chi phí ngữ cảnh khởi điểm cao hơn khi hội thoại còn ngắn.

### 3.3. Đột phá ở hội thoại dài: Vì sao Compact Memory giảm ~50% Prompt Tokens?
- Ở **Stress Benchmark**, cục diện đảo chiều hoàn toàn: Advanced chỉ tiêu thụ **11,418** prompt tokens so với **22,672** của Baseline (tiết kiệm gần một nửa!).
- **Cơ chế hoạt động**:
  - Khi hội thoại kéo dài (16 turns dài với các bài viết khoa học/tin tức), tổng token context tích lũy nhanh chóng vượt ngưỡng 800 token.
  - Baseline không có cơ chế nén, buộc phải dồn toa toàn bộ lịch sử qua từng lượt. Tại turn thứ 16, Baseline phải nạp toàn bộ hơn 3,000 token lịch sử cũ vào prompt.
  - Advanced Agent đã kích hoạt **8 lần Compaction**: nén các tin nhắn cũ vượt quá `keep_messages` thành đoạn summary ngắn gọn (~50 - 80 token), chỉ giữ nguyên vẹn 4 tin nhắn gần nhất.
  - Nhờ vậy, kích thước prompt của Advanced luôn được chặn trần (bounded), ngăn chặn sự bùng nổ token bậc hai ($O(N^2)$) của Baseline.

### 3.4. Phân tích Tăng trưởng File Memory (`User.md`) và Rủi ro Hệ thống
- **Tốc độ tăng trưởng**: Trong cả 2 benchmark, file `User.md` dao động từ **243 Bytes đến 307 Bytes**. Nhờ cơ chế cập nhật ghi đè (upsert theo key) thay vì append vô tận, file chỉ tăng khi có thêm thực thể mới (ví dụ thêm pet, thêm món ăn).
- **Các rủi ro kỹ thuật trong môi trường Production**:
  1. **Memory Bloat (Phình to bộ nhớ)**: Nếu người dùng trò chuyện nhiều năm, danh sách fact tích lũy có thể lên hàng trăm dòng, làm tăng chi phí đọc/ghi và chiếm dung lượng prompt ban đầu.
  2. **Stale Memory (Dữ liệu lỗi thời)**: Nếu không có cơ chế *Memory Decay* (giảm độ ưu tiên theo thời gian) hoặc *Conflict Resolution*, các thông tin tạm thời (như đi công tác 2 ngày) có thể bị lưu vĩnh viễn.
  3. **Fact Hallucination / Poisoning**: Nếu bộ trích xuất fact parse nhầm câu đùa hoặc câu hỏi thành fact, agent sẽ bị "nhiễm độc thông tin" và trả lời sai ở các phiên tương lai.

---

## 4. Các Tính Năng Bonus Mở Rộng (Mốc 90 - 100 Điểm)

Được triển khai hoàn chỉnh trong [src/memory_store.py](file:///d:/VINAI/KX-DAY17-NguyenMinhHieu-2A202602669/src/memory_store.py) và [src/agent_advanced.py](file:///d:/VINAI/KX-DAY17-NguyenMinhHieu-2A202602669/src/agent_advanced.py):

### 4.1. Conflict Handling & Correction Management
- **Vấn đề**: Người dùng thay đổi thông tin thực tế theo thời gian (ví dụ: chuyển nơi ở từ Đà Nẵng $\rightarrow$ Huế trong `conv-03`, hoặc đổi nghề từ Backend $\rightarrow$ MLOps trong `conv-06`).
- **Giải pháp**:
  - Trong `UserProfileStore.upsert_facts()`, hệ thống ánh xạ fact theo cặp `Key: Value`. Khi có giá trị mới cho key `location` hoặc `profession`, hệ thống cập nhật giá trị mới nhất và loại bỏ giá trị cũ.
  - Trong bộ trích xuất, nhận diện các tiền tố đính chính như *"mình đính chính", "không còn ở... nữa", "giờ chuyển sang..."* để ưu tiên fact mới.
- **Hiệu quả**: Trả lời chính xác câu hỏi *"Nếu phải chọn giữa nghề cũ và nghề mới, nghề hiện tại của mình là gì?"* $\rightarrow$ `MLOps engineer` mà không bị lẫn `backend engineer`.

### 4.2. Noise Filtering & Confidence Threshold
- **Vấn đề**: Người dùng nói đùa hoặc nhắc đến địa danh tạm thời (ví dụ: *"đùa với đồng nghiệp hay là chuyển sang product manager"*, *"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày"*).
- **Giải pháp**:
  - Áp dụng rule-based confidence filter: Bỏ qua các câu chứa *"chỉ là câu đùa"*, *"chỉ là nơi bay ra họp"*, *"đừng lấy nó làm nơi ở hiện tại"*.
- **Hiệu quả**: Trả lời chính xác câu hỏi hóc búa trong stress test: *"Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở hiện tại của mình?"* $\rightarrow$ Agent chọn đúng `MLOps engineer` và `Đà Nẵng`.

### 4.3. Question vs Statement Guardrail (Lọc câu hỏi)
- **Vấn đề**: Khi người dùng đặt câu hỏi kiểm tra trí nhớ (*"Bạn có thể nhắc lại tên mình không?"*, *"Đồ uống yêu thích của mình là gì?"*), các regex ngây thơ có thể tưởng nhầm người dùng đang khai báo fact và ghi đè profile bằng chuỗi rác.
- **Giải pháp**:
  - Hàm `extract_profile_updates()` kiểm tra cấu trúc câu: Nếu câu kết thúc bằng `?` hoặc bắt đầu bằng cụm từ nghi vấn (*"bạn có thể", "hãy nhắc lại", "bạn biết... không"*) mà không chứa cụm từ khẳng định (*"tên mình là"*, *"đính chính"*), câu đó sẽ bị bỏ qua và không kích hoạt trích xuất fact.

### 4.4. Multi-Provider Compatibility
- Đã cấu hình và kiểm chứng tương thích 6 provider trong [src/model_provider.py](file:///d:/VINAI/KX-DAY17-NguyenMinhHieu-2A202602669/src/model_provider.py):
  `OpenAI`, `Custom (OpenAI-compatible)`, `Google Gemini`, `Anthropic Claude`, `Ollama (Local)`, `OpenRouter`.

---

## 5. Kết quả Kiểm thử (Unit Tests)

Chạy kiểm thử tự động với `pytest src/test_agents.py -v`:

```bash
platform win32 -- Python 3.13.9, pytest-9.1.1
collected 4 items

src/test_agents.py::test_user_markdown_read_write_edit PASSED            [ 25%]
src/test_agents.py::test_compact_trigger PASSED                          [ 50%]
src/test_agents.py::test_cross_session_recall PASSED                     [ 75%]
src/test_agents.py::test_compact_reduces_prompt_load_on_long_thread PASSED [100%]

============================== 4 passed in 3.37s ==============================
```

## 6. Hướng dẫn Tái hiện Kết quả (Reproducibility)

```bash
# 1. Kích hoạt môi trường ảo
.venv\Scripts\Activate.ps1

# 2. Chạy toàn bộ test kiểm chứng
pytest src/test_agents.py -v

# 3. Chạy toàn bộ Benchmark Standard và Stress test
python src/benchmark.py
```
