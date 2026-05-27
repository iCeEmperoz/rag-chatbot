# 02_rag_chain.py
# Chạy RAG Chatbot bằng Terminal với Gemini (Cơ chế fallback BM25 Local if 429/Exhausted)

import os
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Tắt cảnh báo không cần thiết từ thư viện LangChain để giữ Terminal sạch đẹp
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# ── Cấu hình ──────────────────────────────────────────────────
load_dotenv()

CHROMA_PATH = Path("chroma_db")
CACHE_PATH  = Path("data/chunks_cache.json")

# Khởi tạo các biến hệ thống
vectordb = None
local_chunks = []
use_local_bm25 = False

# Thử tải ChromaDB trước
if CHROMA_PATH.exists():
    try:
        print("🔄 Đang thử tải Vector Database ChromaDB...")
        # Khởi tạo mô hình Embedding với cơ chế Fallback
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            embeddings.embed_query("test")
        except Exception as e:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2"
            )
            
        vectordb = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )
        # Kiểm tra nếu database rỗng (ví dụ: bị crash khi index)
        if len(vectordb.get()['ids']) == 0:
            print("⚠️ Database ChromaDB hiện tại trống rỗng.")
            use_local_bm25 = True
        else:
            print("✓ Tải database ChromaDB thành công!")
    except Exception as e:
        print(f"⚠️ Không thể tải ChromaDB ({e}).")
        use_local_bm25 = True
else:
    print("⚠️ Thư mục database ChromaDB không tồn tại.")
    use_local_bm25 = True

# Nếu không thể dùng ChromaDB, thử tải cache BM25 cục bộ
if use_local_bm25:
    if CACHE_PATH.exists():
        print("🔄 Đang chuyển sang sử dụng Local BM25 Retriever từ cache local...")
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            local_chunks = []
            for item in cached_data:
                local_chunks.append(Document(
                    page_content=item["page_content"],
                    metadata=item["metadata"]
                ))
            print(f"✓ Nạp thành công {len(local_chunks)} chunks từ cache local!")
        except Exception as e:
            print(f"❌ Không thể đọc file cache local tại {CACHE_PATH}: {e}")
            print("👉 Vui lòng chạy file '01_indexing.py' để nạp dữ liệu trước.")
            exit(1)
    else:
        print(f"❌ Không tìm thấy database ChromaDB lẫn file cache local tại {CACHE_PATH}!")
        print("👉 Vui lòng chạy file '01_indexing.py' trước để học/nạp dữ liệu.")
        exit(1)

# Khởi tạo LLM Gemini
print("🔄 Đang kết nối với Google Gemini API...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0  # Đặt nhiệt độ = 0 để giảm thiểu tối đa sự sáng tạo của AI
)

# ── Định nghĩa prompt nghiêm ngặt phòng chống ảo tưởng ──────────────
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp phục vụ việc hỏi đáp tri thức doanh nghiệp.
Nhiệm vụ của bạn là trả lời các câu hỏi dựa TRỰC TIẾP và CHỈ dựa trên phần "Bối cảnh tài liệu" được cung cấp dưới đây.

HÃY TUÂN THỦ NGHIÊM NGẶT CÁC NGUYÊN TẮC SAU:
1. Chỉ sử dụng thông tin được cung cấp trực tiếp trong phần "Bối cảnh tài liệu" để trả lời.
2. Nếu bối cảnh KHÔNG chứa đủ thông tin để trả lời câu hỏi, bạn PHẢI trả lời chính xác là: "Tôi không biết thông tin này trong tài liệu được cung cấp." và không cố gắng suy diễn hay bịa đặt.
3. Tuyệt đối không tự suy diễn, không đoán mò và không sử dụng kiến thức bên ngoài của bạn.
4. Trả lời một cách ngắn gọn, rõ ràng, tập trung thẳng vào câu hỏi.

Bối cảnh tài liệu:
----------------------
{context}
----------------------

Hãy trả lời câu hỏi sau bằng Tiếng Việt."""

# ── Hàm xử lý truy vấn RAG ────────────────────────────────────
def query_rag(question: str):
    results = []
    
    # 1. Retrieval
    if not use_local_bm25 and vectordb is not None:
        try:
            results = vectordb.similarity_search(question, k=4)
        except Exception as e:
            print(f"⚠️ Lỗi tìm kiếm ChromaDB: {e}. Hệ thống tự động chuyển sang BM25 cục bộ cho câu hỏi này...")
            # Fallback sang BM25 trong phiên làm việc
            if CACHE_PATH.exists() and not local_chunks:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                for item in cached_data:
                    local_chunks.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
            
            if local_chunks:
                retriever = BM25Retriever.from_documents(local_chunks)
                retriever.k = 4
                results = retriever.invoke(question)
    else:
        if local_chunks:
            retriever = BM25Retriever.from_documents(local_chunks)
            retriever.k = 4
            results = retriever.invoke(question)
            
    if not results:
        print("⚠️ Không tìm thấy đoạn văn bản liên quan nào trong tài liệu.")
        return
        
    # 2. Xây dựng bối cảnh từ các tài liệu được tìm thấy
    context_parts = []
    citations = []
    
    for i, doc in enumerate(results):
        doc_type = doc.metadata.get("type", "unknown")
        source   = doc.metadata.get("source", "unknown")
        
        # Định nghĩa thông tin trích dẫn chi tiết
        if doc_type == "PDF":
            page = doc.metadata.get("page", 1)
            citation = f"PDF: {source} (Trang {page})"
        elif doc_type == "CSV":
            row = doc.metadata.get("row", 1)
            citation = f"CSV: {source} (Dòng {row})"
        elif doc_type == "DOCX":
            chunk_num = doc.metadata.get("chunk", 1)
            citation = f"Word: {source} (Phần {chunk_num})"
        else:
            citation = f"{source}"
            
        citations.append(citation)
        
        # Thêm vào bối cảnh
        context_parts.append(f"[{citation}]\n{doc.page_content}\n")
        
    context = "\n".join(context_parts)
    
    # 3. Kết hợp System Prompt + Bối cảnh + Câu hỏi
    prompt = SYSTEM_PROMPT.format(context=context)
    
    messages = [
        ("system", prompt),
        ("human", question)
    ]
    
    # 4. Gọi LLM
    try:
        response = llm.invoke(messages)
        answer = response.content
        answer_mode = "Gemini LLM"
    except Exception as e:
        # Xử lý lỗi API Key hết hạn / 429 RESOURCE_EXHAUSTED
        answer_mode = "Hệ thống Cục bộ (Do Gemini API Rate-Limit)"
        answer = (
            f"⚠️ [Lỗi API Gemini - Quota Exhausted / Rate Limit] Không thể gửi yêu cầu tới mô hình ngôn ngữ Google Gemini ({e}).\n"
            "Dưới đây là các đoạn thông tin chính xác được trích lục trực tiếp từ tài liệu của bạn cho câu hỏi này:"
        )
        for idx, doc in enumerate(results):
            answer += f"\n\n👉 Đoạn trích {idx+1} [{citations[idx]}]:\n{doc.page_content.strip()}"
            
    # 5. Hiển thị kết quả ra Terminal
    print("\n" + "="*30 + f" TRẢ LỜI CỦA AI ({answer_mode}) " + "="*30)
    print(answer)
    print("\n" + "-"*30 + " NGUỒN TRÍCH DẪN CHI TIẾT " + "-"*30)
    
    # Loại bỏ các nguồn trùng lặp nếu có
    unique_citations = list(dict.fromkeys(citations))
    for citation in unique_citations:
        print(f"📍 {citation}")
    print("="*76)

# ── Vòng lặp Chat Terminal chính ──────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 RAG CHATBOT - HỎI ĐÁP TÀI LIỆU ĐA ĐỊNH DẠNG 🤖")
    print("="*60)
    print(f"Chế độ tìm kiếm hiện tại: {'ChromaDB (Dense)' if not use_local_bm25 else 'BM25 local (Sparse)'}")
    print("Các định dạng hỗ trợ: PDF, Word (DOCX), Bảng tính (CSV)")
    print("Gõ 'exit' or 'quit' để thoát chương trình.\n")
    
    while True:
        try:
            user_input = input("❓ Nhập câu hỏi của bạn: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 Tạm biệt!")
                break
            if not user_input.strip():
                continue
                
            query_rag(user_input)
            print()
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi: {e}\n")
