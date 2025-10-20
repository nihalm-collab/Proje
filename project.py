import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain bileşenleri
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# -----------------------------------------------------------
# 1. ORTAM AYARLARI VE SABİT TANIMLAMALAR
# -----------------------------------------------------------

# .env dosyasını yükle (Lokalde çalışırken API anahtarını alır)
load_dotenv()

# API Anahtarını kontrol et ve ayarla (Kaggle/Colab'de zaten ortam değişkeni olarak bulunur)
if "GEMINI_API_KEY" not in os.environ:
    st.error("Lütfen GEMINI_API_KEY ortam değişkenini ayarlayın (Lokalde .env, Kaggle'da Secrets).")
    st.stop()

# Model ve Veri Tabanı Sabitleri
VECTOR_DB_DIR = "chroma_db"
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001" # Google'ın önerdiği Embedding modeli

# -----------------------------------------------------------
# 2. VERİ HAZIRLIĞI VE METİNLEŞTİRME FONKSİYONLARI
# -----------------------------------------------------------

def load_data_and_create_documents():
    """
    Kaggle CSV dosyasını yükler, her bir satırı RAG için uygun metin belgesine dönüştürür.
    """
    try:
        # Kaggle veri setinin yolu (Ortama göre yolu güncelleyebilirsiniz)
        data_path = "earthquake_data_1914_2023.csv"
        
        # Veri setini pandas ile oku
        df = pd.read_csv(data_path)
        
        # Sadece kritik sütunları al
        df = df[['Time', 'Latitude', 'Longitude', 'Magnitude', 'Depth/Km', 'Region', 'City']]
        
        # Her bir satırı metin belgesine dönüştürme fonksiyonu
        def create_document_content(row):
            return (
                f"Tarih ve Saat: {row['Time']}. Bölge/Şehir: {row['Region']} / {row['City']}. "
                f"Büyüklük: {row['Magnitude']} şiddetinde. Derinlik: {row['Depth/Km']} km. "
                f"Koordinatlar: {row['Latitude']} enlem, {row['Longitude']} boylam."
            )

        # Tüm DataFrame'i LangChain Document nesnelerine dönüştür (Metinleştirme)
        # Her bir satır bir LangChain Document'ı olacaktır.
        documents = []
        for index, row in df.iterrows():
             documents.append({
                 "page_content": create_document_content(row),
                 "metadata": {"source": f"Kayıt {index+1}"}
             })

        # LangChain'in DataFrameLoader'ı metin içeriğini `page_content` anahtarı ile bekler.
        # Bu yaklaşım, doküman metadata'sını daha iyi kontrol etmemizi sağlar.
        # DataFrameLoader yerine manuel listeyi kullanıyoruz:
        docs = [
            {"page_content": create_document_content(row), "metadata": {"source": f"Kayıt {index+1}", "Region": row['Region'], "Magnitude": row['Magnitude']}}
            for index, row in df.iterrows()
        ]
        
        # Doküman içeriği doğru bir şekilde LangChain Document objesine dönüştürülüyor
        from langchain.schema import Document
        documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]

        return documents

    except FileNotFoundError:
        st.error(f"Veri dosyası bulunamadı: {data_path}. Kaggle ortamında veya yerelde dosyanın adını kontrol edin.")
        return []
    except Exception as e:
        st.error(f"Veri yükleme veya dönüştürme hatası: {e}")
        return []

# -----------------------------------------------------------
# 3. RAG PIPELINE KURULUMU VE İNDEKLEME
# -----------------------------------------------------------

@st.cache_resource
def initialize_rag_pipeline():
    """
    RAG zincirini (Embedding, Vector Store, Retriever, LLM) kurar.
    Streamlit'in cache_resource dekoratörü ile veritabanının sadece bir kez oluşturulması sağlanır.
    """
    st.write("RAG Sistemi Başlatılıyor: Veri Yükleme ve İndeksleme...")

    # 1. Veri Yükleme ve Metinleştirme
    documents = load_data_and_create_documents()
    if not documents:
        return None

    # 2. Parçalama (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # 3. Embedding Model
    # Google'ın Embedding modelini kullan
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # 4. Vektör Depolama (ChromaDB) ve İndeksleme
    # ChromaDB'yi yerel diskte (veya Colab/Kaggle'da oturum belleğinde) oluştur
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_DIR
    )
    
    # 5. Retriever (Geri Çağırma)
    # En alakalı 3 dokümanı çekecek şekilde ayarla
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 6. LLM (Generation Model)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1) # Düşük sıcaklık ile veriye sadık kalma
    
    # 7. Prompt Mühendisliği: Halüsinasyonu önlemek için kritik prompt
    system_prompt = (
        "Sen, Türkiye Deprem Verileri (1914-2023) hakkında uzmanlaşmış bir bilgi asistanısın. "
        "Lütfen **SADECE** sağlanan bağlam (context) içinde yer alan deprem kayıtlarına göre cevap ver. "
        "Eğer bilgi bağlamda yoksa, 'Bu bilgi veri setinde bulunmamaktadır.' diye cevapla. "
        "Cevaplarını düzenli ve bilgilendirici bir dille ver. "
        "\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # 8. RAG Zinciri Oluşturma (LangChain)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    st.write(f"RAG Sistemi Başlatıldı! {len(splits)} adet metin parçası indekslendi.")
    return retrieval_chain

# -----------------------------------------------------------
# 4. STREAMLIT WEB ARAYÜZÜ (app.py'nin Ana Fonksiyonu)
# -----------------------------------------------------------

def main():
    """Streamlit uygulamasını çalıştıran ana fonksiyon."""
    st.title("🇹🇷 Deprem Verileri Bilgi Asistanı (RAG)")
    
    # RAG Pipeline'ı başlat ve önbelleğe al
    rag_chain = initialize_rag_pipeline()

    if rag_chain is None:
        st.stop()

    # Sohbet geçmişini başlat (Streamlit Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Başlangıç mesajını ekle
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Merhaba! Türkiye Deprem Verileri (1914-2023) hakkında dilediğiniz soruyu sorabilirsiniz. Veri setine sadık kalarak cevap vereceğim."})

    # Sohbet geçmişini göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if prompt := st.chat_input("Sorunuzu buraya yazın (Örn: En büyük deprem ne zaman oldu?)"):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chatbot yanıtını üret
        with st.chat_message("assistant"):
            with st.spinner("Yanıt aranıyor..."):
                try:
                    # RAG zincirini çağır
                    response = rag_chain.invoke({"input": prompt})
                    
                    # Cevabı al
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Kullanılan kaynakları (opsiyonel) göster
                    sources = [doc.metadata.get("source", "Bilinmeyen Kaynak") for doc in response["context"]]
                    if sources:
                        unique_sources = list(set(sources))
                        st.caption(f"Kullanılan Kaynaklar: {', '.join(unique_sources)}")
                        
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"Bir hata oluştu: {e}. Lütfen API anahtarınızı ve bağlantıyı kontrol edin."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
