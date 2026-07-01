# BÀI VIẾT NGẮN ĐI KÈM (WRITE-UP) — LAB 25: GPU FINOPS OPTIMIZATION
**Họ và tên:** Mai Văn Thuyên  
**Mã sinh viên:** 2A202600926  

---

## 1. Baseline vs. Optimized (So sánh trước và sau tối ưu)

Dựa trên kết quả chạy từ hệ thống kiểm thử tự động và các mission, chi phí và hiệu năng sử dụng tài nguyên của NimbusAI đã được cải thiện rõ rệt:

*   **Chi phí hàng tháng (Spend):**
    *   **Baseline Spend (Chi phí gốc):** `$27,133`
    *   **Optimized Spend (Chi phí sau tối ưu):** `$14,626`
    *   **Projected Savings (Tiết kiệm dự kiến):** `$12,507` (tương đương với giảm **46%** tổng chi phí).
*   **Đơn giá trên lượng dữ liệu xử lý ($/1M-token):**
    *   **Baseline:** `$36.643 / 1M-token`
    *   **Optimized:** `$10.374 / 1M-token`
    *   **Hiệu quả:** Tiết kiệm được **71.7%** đơn giá mỗi triệu token phục vụ cho hệ thống inference nhờ các kỹ thuật tối ưu hóa routing.

---

## 2. Phân tích các đòn bẩy tiết kiệm (FinOps Levers Breakdown)

Dưới đây là bảng chi tiết các đòn bẩy đóng góp vào tổng khoản tiết kiệm hằng tháng:

| Thứ tự | Đòn bẩy tối ưu | Khoản tiết kiệm hàng tháng (USD) | Tỷ lệ đóng góp (%) |
|---|---|---|---|
| 1 | **Purchasing (spot/reserved)** | `$10,040` | **80.3%** |
| 2 | **Inference (cascade/cache/batch)** | `$1,212` | **9.7%** |
| 3 | **Right-size util-lies** | `$655` | **5.2%** |
| 4 | **Kill idle GPUs** | `$600` | **4.8%** |
| | **Tổng cộng** | **$12,507** | **100%** |

### Nhận xét & Phân tích chi tiết:
*   **Đòn bẩy đóng góp lớn nhất:** **Purchasing Strategy (Spot & Reserved Instances)** đóng góp tới **80.3%** tổng lượng tiền tiết kiệm.
    *   *Tại sao?* Vì giá thuê GPU On-Demand cực kỳ đắt đỏ. Bằng cách dịch chuyển các training workloads có khả năng checkpoint và chịu được gián đoạn sang **Spot** (giảm ~60% chi phí) và cam kết **Reserved 3yr** đối với các inference workload chạy liên tục 24/7 (duty cycle cao hơn điểm hòa vốn 55%), ta đã cắt giảm lượng lớn chi phí lãng phí cố định.
*   **Inference Optimization** (cascade, prompt caching, batching) mang lại savings rất lớn cho dịch vụ API. Việc định tuyến sang các model nhỏ rẻ hơn 15 lần cho các task đơn giản (Cascade) kết hợp với chiết khấu 90% cho input cache và 50% cho offline batch jobs đã làm sụt giảm đơn giá $/1M-token từ `$36.64` xuống chỉ còn `$10.37`.

---

## 3. GPU-Util Lie (Sự lừa dối từ chỉ số GPU-Util %)

Trong quá trình kiểm toán tại **Mission 1**, hệ thống đã phát hiện hai GPU nằm trong diện "GPU-Util Lie" (có chỉ số Utilization đo từ `nvidia-smi` cực kỳ cao nhưng MFU/MBU thực tế lại rất thấp):
1.  **`gpu-h100-4`**: Có `GPU-Util` lên tới **98%** nhưng MFU thực tế chỉ đạt **20.2%** (0.202).
2.  **`gpu-a10g-1`**: Có `GPU-Util` ở mức cao nhưng MFU thực tế thấp dưới 30%.

### Giải thích bản chất kỹ thuật:
*   **Chỉ số GPU-Util % từ nvidia-smi** chỉ phản ánh tỷ lệ thời gian mà các nhân đồ họa hoặc bộ điều khiển bộ nhớ (memory controller) có phát sinh hoạt động (tức là clock đang chạy tích cực, lớn hơn 0). Nó hoàn toàn **không đo lường hiệu năng tính toán thực tế**.
*   **Tại sao GPU-Util 98% nhưng MFU chỉ có 20%?** Hiện tượng này thường xảy ra do:
    1.  **Memory Stall (Nghẽn băng thông bộ nhớ):** GPU phải dành phần lớn thời gian chờ dữ liệu truyền từ bộ nhớ HBM sang thanh ghi phục vụ tính toán (Memory-bound). Điều này cực kỳ phổ biến trong LLM autoregressive decode phase.
    2.  **Kernel Launch Overhead / Batch Size quá nhỏ:** Lượng tính toán trong mỗi lần gọi kernel quá nhỏ làm cho thời gian chuẩn bị dữ liệu (CPU-to-GPU overhead) chiếm chủ đạo, khiến GPU bận rộn một cách vô ích.
*   **Tác động tài chính:** Ta đang phải trả 100% đơn giá thuê theo giờ đối với dòng GPU H100 cao cấp nhưng chỉ tận dụng được 20% năng lực tính toán lý thuyết của nó. Giải pháp xử lý là hạ cấp (Right-size) các GPU này xuống các dòng rẻ hơn (như A100 hoặc A10G) hoặc gộp batch để tăng MFU.

---

## 4. Mô tả các phần mở rộng đã thực hiện (Extensions)

Chúng tôi đã hoàn thành xuất sắc **2 extensions** để cải thiện độ thông minh của hệ thống FinOps:

### Extension 3: Xác định tính kinh tế của Prompt Caching (`cache_is_worth_it()`)
*   **Mô tả:** Prompt Caching của các nhà cung cấp như Gemini hay Anthropic đòi hỏi chi phí ghi cache (write cost) ban đầu và chỉ giảm giá cho các lượt đọc lại sau đó (read discount = 90%, tức chỉ tính 10% giá gốc). Nếu prompt chỉ được đọc 1 lần rồi bỏ, việc ghi cache sẽ gây lỗ tiền.
*   **Logic:** Hàm `cache_is_worth_it()` đánh giá điều kiện hòa vốn: 
    $$\text{avg\_cache\_reads} \times (1 - \text{read\_discount}) > 1.0 \implies \text{avg\_cache\_reads} \times 0.9 > 1.0 \implies \text{avg\_cache\_reads} > 1.11$$
*   **Kết quả:** Với số lượt đọc trung bình thực tế của hệ thống là `4.0` (> 1.11), việc áp dụng prompt caching hoàn toàn khả thi và đã được kích hoạt trong tối ưu hóa M2, giúp giảm chi phí đầu vào hiệu quả.

### Extension 5: Carbon-aware Scheduling (Lập lịch thông minh giảm phát thải)
*   **Mô tả:** Các công việc training không yêu cầu real-time và có thể gián đoạn được lập lịch dịch chuyển từ vùng có lưới điện phát thải cao sang vùng có lưới điện sạch (sử dụng năng lượng tái tạo).
*   **Kết quả đo lường cụ thể:**
    *   Chuyển các interruptible jobs từ vùng **`us-east-1`** (chủ yếu là điện than, phát thải **380 gCO2/kWh**) sang vùng **`europe-north1`** (Na Uy, sử dụng thủy điện sạch, phát thải chỉ **30 gCO2/kWh**).
    *   Tổng lượng năng lượng tiêu thụ cho các job này là **4,351.3 kWh**.
    *   Khi chạy ở `us-east-1`, lượng phát thải là **1,653.5 kg CO2**. Khi chuyển sang `europe-north1`, lượng phát thải giảm chỉ còn **130.5 kg CO2**.
    *   Tiết kiệm được **1,523.0 kg CO2** (tương đương **1,522,965 gCO2e**, giảm đến **92.1%** lượng carbon phát thải) trong khi chi phí điện năng cũng được tối ưu tương ứng.

---

## 5. Khuyến nghị cho NimbusAI (Action Plan)

Nếu đảm nhiệm vai trò FinOps Lead tại NimbusAI, 3 hành động đầu tiên tôi sẽ thực hiện ngay lập tức gồm:

1.  **Áp dụng ngay Tagging Policy & Kích hoạt Chargeback:**
    *   Hiện tại Tag Coverage đã đạt **92%** (vượt ngưỡng an toàn 80%). Tôi sẽ chính thức mở cổng **Chargeback**, chuyển từ Showback (chỉ hiển thị) sang trừ tiền trực tiếp vào ngân sách của các đội (đặc biệt là đội `ml-research` vốn đang tiêu tốn nhiều chi phí nhất). Điều này tạo động lực tài chính trực tiếp bắt buộc các kỹ sư phải tự tối ưu hóa code.
2.  **Chuyển đổi hình thức mua GPU (Purchasing Migration):**
    *   Dừng ngay việc thuê On-demand cho các GPU chạy liên tục. Lập lịch tự động chuyển các workloads huấn luyện sang **Spot Instance** đi kèm cơ chế checkpoint tự động, đồng thời mua cam kết dài hạn **Reserved Instances 3 năm** cho các GPU phục vụ APIs sản xuất chính.
3.  **Tách cấu trúc Prefill và Decode (Disaggregation) & Áp dụng Prompt Caching:**
    *   Khắc phục triệt để lỗi "GPU-Util Lie" trên H100 bằng cách chia nhỏ quá trình suy luận. Prefill (Compute-bound) sẽ được chạy trên GPU mạnh như H100 để trả kết quả nhanh, còn Decode (Memory-bound) hoặc các request đơn giản sẽ được định tuyến sang dòng GPU rẻ hơn như A10G/L4 kết hợp cơ chế gộp batch và Prompt Caching để tăng hiệu suất sử dụng băng thông bộ nhớ (MBU).
