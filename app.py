# app.py
# Streamlit Web UI - RAG Chatbot Tri Thức Doanh Nghiệp Đa Định Dạng

import os
import json
import shutil
import warnings
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

load_dotenv()

DATA_DIR    = Path("data")
CHROMA_PATH = Path("chroma_db")
CACHE_PATH  = Path("data/chunks_cache.json")

DATA_DIR.mkdir(exist_ok=True)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Multi-Format RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Cấu trúc font toàn hệ thống */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Ẩn thanh header và nút Deploy mặc định của Streamlit */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Nền chính màu Dark Slate tối giản kiểu Claude/OpenAI */
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #111827 100%);
        color: #F3F4F6;
    }
    
    /* Thiết kế Sidebar giảm saturation, mượt mà tinh tế */
    section[data-testid="stSidebar"] {
        background: rgba(11, 15, 25, 0.65) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }
    
    /* Đẩy toàn bộ nội dung của Sidebar lên sát mép trên */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.5rem !important;
        margin-top: -3.5rem !important;
    }
    
    /* Tiêu đề sidebar tối giản */
    .sidebar-title {
        font-weight: 700;
        background: linear-gradient(to right, #60A5FA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        margin-bottom: 25px;
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* Cấu hình lại các sidebar labels */
    section[data-testid="stSidebar"] h3 {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #9CA3AF !important;
        margin-bottom: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Card Glassmorphic cho tệp tin - Dạng Compact */
    .file-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .file-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Icon định dạng file */
    .file-icon {
        font-size: 16px;
        margin-right: 8px;
    }
    .file-pdf { color: #EF4444; }
    .file-docx { color: #3B82F6; }
    .file-csv { color: #10B981; }
    
    .file-name {
        font-size: 13px !important;
        font-weight: 400 !important;
        color: #E5E7EB;
    }
    
    /* Khung chat và bong bóng chat dạng Premium Product */
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 12px;
        max-width: 82%;
        line-height: 1.7 !important;
        font-size: 15.5px;
    }
    
    .chat-user {
        background: #2563EB;
        color: #ffffff;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .chat-bot {
        background: #1E293B;
        color: #F3F4F6;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Nút kích hoạt học dữ liệu dạng Premium */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.2) !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(29, 78, 216, 0.35) !important;
    }
    
    /* CSS Suggestion Chips */
    .stButton>button.suggestion-chip {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        color: #9CA3AF !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        box-shadow: none !important;
        width: auto !important;
    }
    .stButton>button.suggestion-chip:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: #3B82F6 !important;
        color: #ffffff !important;
    }
    
    /* Source Card hiển thị nguồn kiểu Perplexity/Claude */
    .citation-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: -6px;
        margin-bottom: 24px;
        padding-left: 2px;
    }
    .citation-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 6px 12px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12.5px;
        color: #9CA3AF;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .citation-card:hover {
        background: rgba(59, 130, 246, 0.05);
        border-color: rgba(59, 130, 246, 0.3);
        color: #60A5FA;
    }
    
    /* Expander trích dẫn chi tiết */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        color: #9CA3AF !important;
    }
    
    /* Tiêu đề chính */
    .main-title {
        font-weight: 700;
        background: linear-gradient(to right, #FFFFFF 40%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 38px;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    /* Status Indicator Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 15px;
    }
    .status-badge.ready {
        background: rgba(16, 185, 129, 0.08);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.15);
    }
    .status-badge.processing {
        background: rgba(245, 158, 11, 0.08);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.15);
    }
    .status-badge.failed {
        background: rgba(239, 68, 68, 0.08);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.15);
    }
    
    /* Tắt hoàn toàn hiệu ứng tối/mờ toàn bộ trang khi đang xử lý */
    div[data-testid="stAppViewBlockContainer"], 
    div[data-testid="stAppViewBlockContainer"] > div,
    [data-testid="stHeader"],
    div.element-container,
    div[class*="st-emotion-cache"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
    
</style>
""", unsafe_allow_html=True)

# ── Hàm Helper nạp và xử lý tài liệu ───────────────────────────
@st.cache_resource
def initialize_system_components():
    """Tải database Chroma hoặc BM25 cache"""
    vectordb = None
    local_chunks = []
    use_local_bm25 = False
    
    # 1. Thử tải ChromaDB
    if CHROMA_PATH.exists():
        try:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                embeddings.embed_query("test")
            except Exception:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
            
            vectordb = Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=embeddings
            )
            if len(vectordb.get()['ids']) == 0:
                use_local_bm25 = True
        except Exception:
            use_local_bm25 = True
    else:
        use_local_bm25 = True
        
    # 2. Thử tải BM25
    if use_local_bm25 and CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            local_chunks = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in cached_data
            ]
        except Exception:
            pass
            
    return vectordb, local_chunks, use_local_bm25

@st.cache_resource
def load_llm():
    """Tải và khởi tạo mô hình ngôn ngữ lớn LLM"""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0
        )
    except Exception:
        return None

# ── Gọi hàm khởi tạo Components ───────────────────────────────
vectordb, local_chunks, use_local_bm25 = initialize_system_components()
llm = load_llm()

# Hệ thống nhắc nhở strict phòng ảo tưởng
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

# Khởi tạo trạng thái hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexing_status" not in st.session_state:
    st.session_state.indexing_status = "ready"
if "temp_query" not in st.session_state:
    st.session_state.temp_query = None

# ── GIAO DIỆN SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 RAG CHATBOT</div>', unsafe_allow_html=True)
    
    st.markdown("### 📥 Nạp Tài Liệu (PDF/DOCX/CSV)")
    uploaded_files = st.file_uploader(
        "Kéo thả hoặc chọn tệp tin (PDF, DOCX, CSV)", 
        type=["pdf", "docx", "csv"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Nút bấm kích hoạt học dữ liệu
    if st.button("⚡ KÍCH HOẠT HỌC DỮ LIỆU"):
        if uploaded_files:
            st.session_state.indexing_status = "processing"
            with st.spinner("🔄 Hệ thống đang đọc tài liệu và chuyển đổi cấu trúc..."):
                # 1. Lưu các file upload vào data/
                for f in uploaded_files:
                    target_file = DATA_DIR / f.name
                    with open(target_file, "wb") as buffer:
                        shutil.copyfileobj(f, buffer)
                
                # 2. Gọi script 01_indexing.py để build database
                try:
                    import subprocess
                    result = subprocess.run(
                        [".\\venv\\Scripts\\python.exe", "01_indexing.py", "--no-test"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8"
                    )
                    
                    # Cập nhật components trong session (xóa cache trước khi nạp lại)
                    initialize_system_components.clear()
                    vectordb, local_chunks, use_local_bm25 = initialize_system_components()
                    st.session_state.indexing_status = "ready"
                    st.success("✅ Hệ thống đã hoàn tất học dữ liệu đa định dạng!")
                    st.toast("Bộ nhớ Vector đã được làm mới thành công!")
                except Exception as e:
                    st.session_state.indexing_status = "failed"
                    st.error(f"❌ Có lỗi xảy ra trong quá trình học dữ liệu: {e}")
        else:
            st.warning("⚠️ Vui lòng kéo thả ít nhất 1 file để kích hoạt!")
            
    # Hiển thị danh sách file đang có trong cơ sở dữ liệu
    st.markdown("---")
    st.markdown("### 🗂️ Danh Sách Tài Liệu Hiện Có")
    
    # Badge trạng thái hệ thống RAG
    if st.session_state.indexing_status == "ready":
        st.markdown('<div class="status-badge ready">● Hệ thống sẵn sàng</div>', unsafe_allow_html=True)
    elif st.session_state.indexing_status == "processing":
        st.markdown('<div class="status-badge processing">● Đang học dữ liệu...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge failed">● Đồng bộ thất bại</div>', unsafe_allow_html=True)
        
    # Thanh tìm kiếm tài liệu cục bộ
    search_query = st.text_input(
        "🔍 Tìm tài liệu...", 
        key="file_search", 
        label_visibility="collapsed", 
        placeholder="🔍 Tìm tài liệu..."
    )
    
    existing_files = list(DATA_DIR.glob("*.*"))
    # Lọc bỏ file cache json
    existing_files = [f for f in existing_files if f.name != "chunks_cache.json"]
    
    if search_query:
        existing_files = [f for f in existing_files if search_query.lower() in f.name.lower()]
        
    if existing_files:
        for f in existing_files:
            ext = f.suffix.lower()
            if ext == ".pdf":
                icon, cls = "📕", "file-pdf"
            elif ext == ".docx":
                icon, cls = "📘", "file-docx"
            elif ext == ".csv":
                icon, cls = "📗", "file-csv"
            else:
                icon, cls = "📄", ""
                
            st.markdown(f"""
            <div class="file-card">
                <div style="display:flex; align-items:center;">
                    <span class="file-icon {cls}">{icon}</span>
                    <span class="file-name">{f.name[:25] + '...' if len(f.name) > 25 else f.name}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        if search_query:
            st.markdown("<p style='font-style:italic; color:#6B7280; font-size:13px; text-align:center; margin-top:10px;'>Không tìm thấy tài liệu phù hợp.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-style:italic; color:#6B7280; font-size:13px; text-align:center; margin-top:10px;'>Danh sách tài liệu hiện đang trống.</p>", unsafe_allow_html=True)

# ── GIAO DIỆN CHAT CHÍNH ───────────────────────────────────────
st.markdown('<div class="main-title">🧠 Multi-Format RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #a5b4fc; font-size: 16px; margin-top:-5px;">Hỏi đáp tài liệu thông minh, chống ảo tưởng AI và hiển thị nguồn trích dẫn thời gian thực.</p>', unsafe_allow_html=True)
st.markdown("---")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:flex-end;">
            <div class="chat-bubble chat-user">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:flex-start;">
            <div class="chat-bubble chat-bot">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Nếu có nguồn trích dẫn, vẽ expander
        if "citations" in msg and msg["citations"]:
            
            with st.expander("📍 Chi tiết nguồn trích dẫn tài liệu"):
                for cit, text in zip(msg["citations"], msg["raw_texts"]):
                    st.markdown(f"**📍 {cit}**")
                    st.info(text)

# Xử lý nhập câu hỏi mới
if user_query := st.chat_input("Hỏi tôi bất cứ điều gì về tài liệu..."):
    # Hiển thị câu hỏi của user
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:flex-end;">
        <div class="chat-bubble chat-user">
            {user_query}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Tiến hành xử lý RAG
    with st.spinner("🧠 Đang tra cứu tài liệu và suy luận câu trả lời..."):
        results = []
        
        # 1. Retrieval
        if not use_local_bm25 and vectordb is not None:
            try:
                results = vectordb.similarity_search(user_query, k=4)
            except Exception:
                if local_chunks:
                    retriever = BM25Retriever.from_documents(local_chunks)
                    retriever.k = 4
                    results = retriever.invoke(user_query)
        else:
            if local_chunks:
                retriever = BM25Retriever.from_documents(local_chunks)
                retriever.k = 4
                results = retriever.invoke(user_query)
                
        if results:
            context_parts = []
            citations = []
            raw_texts = []
            
            for doc in results:
                doc_type = doc.metadata.get("type", "unknown")
                source = doc.metadata.get("source", "unknown")
                
                if doc_type == "PDF":
                    page = doc.metadata.get("page", 1)
                    citation = f"File PDF: {source} (Trang {page})"
                elif doc_type == "CSV":
                    row = doc.metadata.get("row", 1)
                    citation = f"File CSV: {source} (Dòng {row})"
                elif doc_type == "DOCX":
                    chunk_num = doc.metadata.get("chunk", 1)
                    citation = f"Word DOCX: {source} (Phần {chunk_num})"
                else:
                    citation = f"Nguồn: {source}"
                    
                citations.append(citation)
                raw_texts.append(doc.page_content)
                context_parts.append(f"[{citation}]\n{doc.page_content}\n")
                
            context = "\n".join(context_parts)
            prompt = SYSTEM_PROMPT.format(context=context)
            
            # Gọi LLM Gemini
            messages = [
                ("system", prompt),
                ("human", user_query)
            ]
            
            try:
                if llm is not None:
                    response = llm.invoke(messages)
                    answer = response.content
                else:
                    raise Exception("Chưa cấu hình LLM")
            except Exception as e:
                answer = (
                    "⚠️ **[Giới hạn Quota API Gemini]** API Key hiện tại tạm thời vượt quá giới hạn gọi hoặc chưa hoạt động.\n"
                    "Dưới đây là các đoạn văn bản khớp chính xác nhất được tìm thấy cục bộ trong hệ thống:"
                )
                # Ghép các trích đoạn vào câu trả lời trực tiếp
                for i, (cit, r_txt) in enumerate(zip(citations, raw_texts)):
                    answer += f"\n\n**📍 Trích đoạn {i+1} [{cit}]:**\n>{r_txt.strip()}"
                    
            # Hiển thị câu trả lời của Bot
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; align-items:flex-start;">
                <div class="chat-bubble chat-bot">
                    {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị expander trích dẫn ngay dưới câu trả lời
            with st.expander("📍 Chi tiết nguồn trích dẫn tài liệu"):
                for cit, text in zip(citations, raw_texts):
                    st.markdown(f"**📍 {cit}**")
                    st.info(text)
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "raw_texts": raw_texts
            })
            
        else:
            no_info_msg = "Tôi không tìm thấy bất kỳ tài liệu liên quan nào trong kho dữ liệu của bạn để trả lời câu hỏi này."
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; align-items:flex-start;">
                <div class="chat-bubble chat-bot">
                    {no_info_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": no_info_msg})
            
    # Refresh trang để vẽ lại mượt mà st.chat_input
    st.rerun()
