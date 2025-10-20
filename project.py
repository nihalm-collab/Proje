import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# LangChain/RAG Bileşenleri
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Projenin ana veri yolu ve konfigürasyonları
# NOT: Kaggle'da 'os.getenv' ile secrets/ortam değişkenlerini çekebilirsiniz.
# Lokal çalıştırma için .env dosyası kullanılabilir.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =========================================================================
# Veri Hazırlama Fonksiyonları
# =========================================================================

@st.cache_resource
def load_and_prepare_data():
    """
    Kaggle CSV dosyasını yükler ve her satırı RAG için metin belgesine dönüştürür.
    Bu, tabular veriyi metin tabanlı RAG için uygun hale getirmenin kritik adımıdır.
    """
    try:
        # Kaggle'da veri setine erişim yolu bu formattadır.
        # Kullanıcının veri seti yolunu doğru girdiğinden emin olun.
        data_path = "/kaggle/input/turkey-earthquake-data-1914-2023/earthquake.csv"
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("Veri seti bulunamadı. Lütfen Kaggle ortamında doğru yolu kullandığınızdan emin olun.")
        return []

    documents = []
    for index, row in df.iterrows():
        # Her bir deprem kaydını anlamlı, tek bir metin parçasına dönüştürme
        text_content = (
            f"Tarih: {row['date_']}. Enlem: {row['latitude']:.2f}, Boylam: {row['longitude']:.2f}. "
            f"Şiddet (Magnitude): {row['magnitude']:.1f} ({row['type']}). "
            f"Derinlik: {row['depth']:.1f} km. Bölge: {row['region_']}"
        )
        documents.append(Document(page_content=text_content, metadata={"source": row['date_'], "magnitude": row['magnitude']}))

    st.success(f"{len(documents)} adet deprem kaydı metin belgesine dönüştürüldü.")
    return documents

# =========================================================================
# RAG Pipeline Kurulumu
# =========================================================================

@st.cache_resource
def setup_rag_pipeline(documents):
    """
    RAG mimarisini kurar: Çanklama, Embedding, Vektör DB ve QA Zinciri.
    """
    if not documents:
        return None

    # 1. Metin Parçalama (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    st.info(f"Metinler {len(texts)} parçaya (chunk) bölündü.")

    # 2. Embedding Modeli ve Vektör DB (ChromaDB)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    
    # 3. Retriever: En alakalı 3 belgeyi geri çağırır
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 4. Prompt Şablonu: LLM'i sadece sağlanan bağlama göre cevap vermeye zorlar
    template = """Sen Türkiye'deki deprem verileri hakkında uzman bir bilgi asistanısın.
    Aşağıdaki bağlamı (CONTEXT) kullanarak kullanıcının sorusunu (QUESTION) cevapla.
    Sadece bağlamda bulunan bilgilere dayan. Eğer bağlamda cevap yoksa,
    'Bu soruya elimdeki verilerle cevap veremiyorum.' şeklinde yanıtla.
    Cevabın kapsamlı ve anlaşılır olsun.
    
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 5. RetrievalQA Zinciri
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2), # Düşük temperature halüsinasyonu azaltır
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    st.success("RAG Pipeline başarıyla kuruldu.")
    return qa_chain

# =========================================================================
# Streamlit Arayüzü
# =========================================================================

def main():
    """
    Streamlit arayüzünü oluşturur ve chatbot mantığını yönetir.
    """
    st.title("🇹🇷 Deprem Verileri Bilgi Asistanı (RAG Chatbot)")
    st.caption("Akbank GenAI Bootcamp Projesi | Veri: 1914-2023 Türkiye Deprem Verileri")

    if not GEMINI_API_KEY:
        st.warning("Lütfen `GEMINI_API_KEY` ortam değişkenini/Kaggle Secret'ı ayarlayın.")
        return

    # Veri yükleme ve RAG pipeline kurulumu
    with st.spinner("Veriler yükleniyor ve RAG mimarisi kuruluyor..."):
        documents = load_and_prepare_data()
        qa_chain = setup_rag_pipeline(documents)

    if qa_chain is None:
        st.error("RAG kurulumu başarısız oldu.")
        return

    # Sohbet geçmişini saklama
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Deprem verileri hakkında ne öğrenmek istersiniz? Örneğin: 'En büyük 5 deprem ne zaman oldu?'"}
        ]

    # Sohbet geçmişini gösterme
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Kullanıcı girişi
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # RAG sorgusunu çalıştır
        with st.spinner("Cevap aranıyor..."):
            result = qa_chain({"query": prompt})
            
            # Cevap ve Kaynakları biçimlendir
            response = result["result"]
            sources = result["source_documents"]
            
            # Kaynakları göster
            source_info = "## Kaynaklar:\n"
            for i, doc in enumerate(sources):
                source_info += f"- **{i+1}. Kaynak (Şiddet: {doc.metadata.get('magnitude'):.1f})**: {doc.page_content}\n"

            assistant_response = f"{response}\n\n---\n\n{source_info}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.chat_message("assistant").write(assistant_response)

if __name__ == "__main__":
    main()
