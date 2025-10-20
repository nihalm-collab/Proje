# ==============================================================================
# PROJE ADI: RAG Tabanlı Deprem Bilgilendirme Chatbotu
# AMAÇ: Tablosal deprem verilerinden türetilen metinleri kullanarak kullanıcıya
#       bilgi veren bir RAG chatbotu geliştirmek ve Streamlit ile sunmak.
# ==============================================================================

# ==============================================================================
# BÖLÜM 1: GEREKLİ KÜTÜPHANELERİN VE ORTAM AYARLARININ İÇE AKTARILMASI
# ==============================================================================
import os
from dotenv import load_dotenv
import pandas as pd # CSV dosyasını okumak için
import streamlit as st # Web arayüzü için

# RAG Mimarisi Bileşenleri (LangChain Önerisi)
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma # Vektör veritabanı
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # LLM ve Embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter # Metin parçalama (Chunking)
from langchain.chains import RetrievalQA # RAG Zinciri

# API Anahtarını Yükleme
load_dotenv()
# .env dosyasındaki anahtarı ortam değişkenlerine yükler
# Bu, GEMINI_API_KEY'in doğrudan koda yazılmasını önler.
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY, .env dosyasında tanımlanmalıdır.")

# ==============================================================================
# BÖLÜM 2: VERİ HAZIRLAMA VE DÖNÜŞÜM
# (Knowledge Base Oluşturma)
# ==============================================================================

# Sabitler
CHROMA_PATH = "chroma_db_deprem" # Vektör veritabanının diskte saklanacağı yer
CSV_FILE = "veriler.csv" # Kaggle'dan indirilen tablosal veri

def load_and_transform_data(csv_file):
    """
    Tablosal veriyi okur ve RAG için metinsel formata dönüştürür.
    """
    try:
        df = pd.read_csv(csv_file)
        
        # DataFrame'deki her bir satırı (deprem kaydı) metin formatına dönüştürme (KRİTİK ADIM)
        # RAG, tablosal veriden çok, metinsel bilgilerle daha iyi çalışır.
        # Deprem verilerinden bilgilendirici bir metin türetme:
        df['text'] = df.apply(lambda row: (
            f"Türkiye'de {row['Olus tarihi']} tarihinde, {row['Yer']} bölgesinde "
            f"Moment Büyüklüğü (Mw) {row['Mw']} olarak kaydedilen bir deprem meydana gelmiştir. "
            f"Depremin odak derinliği {row['Der (km)']} km'dir. Enlem: {row['Enlem']}, Boylam: {row['Boylam']}. "
            f"Tip: {row['Tip']}."
        ), axis=1)

        # LangChain'in DataFrameLoader'ı ile her satırı bir "Document" nesnesine dönüştürme
        # Document'in içeriği (page_content) 'text' sütunundan alınır.
        loader = DataFrameLoader(df, page_content_column='text')
        documents = loader.load()
        return documents
    except FileNotFoundError:
        st.error(f"Hata: {csv_file} dosyası bulunamadı. Lütfen dosyanın proje dizininde olduğundan emin olun.")
        return []

# ==============================================================================
# BÖLÜM 3: RAG PIPELINE OLUŞTURMA (Indexing ve Retrieval)
# ==============================================================================

def setup_rag_pipeline(documents):
    """
    Metinleri parçalar (chunk), gömer (embed) ve vektör veritabanını (ChromaDB) kurar.
    """
    if not documents:
        return None, None

    # 3.1. Metin Parçalama (Chunking)
    # RecursiveCharacterTextSplitter, metni ayraçlara göre (örn. \n\n, \n, boşluk) parçalar.
    # Bu, LLM'in dikkatini küçük, anlamlı parçalara odaklamasını sağlar.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Her parçanın maksimum karakter boyutu (Örnek Değer)
        chunk_overlap=200 # Parçaların birbiriyle ne kadar örtüşeceği (Bağlam koruması için)
    )
    # Parçalanan dokümanlar
    chunks = text_splitter.split_documents(documents)
    
    # 3.2. Gömme Modeli (Embedding Model)
    # Metinleri sayısal vektörlere dönüştürmek için kullanılır.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Google'ın önerilen embedding modeli
    
    # 3.3. Vektör Veritabanına Kaydetme (ChromaDB)
    # Vektörler, diske kaydedilmek üzere ChromaDB'ye eklenir. 
    # Bu adım, her çalıştırmada tekrar yapılmaz; yalnızca ilk kurulumda yapılır.
    if not os.path.exists(CHROMA_PATH):
        st.info("ChromaDB oluşturuluyor... (Bu işlem biraz zaman alabilir)")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH # Veritabanını kaydet
        )
        db.persist()
        st.success("ChromaDB başarıyla oluşturuldu ve kaydedildi.")
    else:
        # Daha önce oluşturulmuş veritabanını yükle
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        st.info("Mevcut ChromaDB yüklendi.")
        
    return db

def create_qa_chain(db):
    """
    Generative Model (LLM) ile Retrieval (Geri Getirme) aracını birleştiren zinciri oluşturur.
    """
    # 3.4. Geri Getirme Aracı (Retriever)
    # Vektör DB'de arama yapar (semantic search).
    retriever = db.as_retriever(search_kwargs={"k": 3}) # En alakalı 3 parçayı getir
    
    # 3.5. LLM (Generative Model) Ayarı
    # ChatGoogleGenerativeAI ile Gemini modelini kullanma
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    
    # 3.6. RAG Zinciri (RetrievalQA Chain)
    # Retriever'dan gelen bağlamı LLM'e vererek yanıt üretme.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Geri getirilen tüm parçaları tek prompt'a ekler
        retriever=retriever,
        return_source_documents=True # Cevabın hangi kaynaktan geldiğini görmek için
    )
    
    return qa_chain

# ==============================================================================
# BÖLÜM 4: WEB ARAYÜZÜ VE CHATBOT MANTIĞI (STREAMLIT)
# ==============================================================================

def main():
    """
    Streamlit uygulamasının ana fonksiyonu.
    """
    # 4.1. Arayüz Ayarları ve Sistem Başlığı
    st.set_page_config(page_title="Deprem Bilgilendirme RAG Chatbotu")
    st.title("Türkiye Deprem Bilgilendirme Asistanı 🌍")
    st.caption("RAG (Retrieval Augmented Generation) ile Güçlendirilmiştir")
    
    # 4.2. RAG Pipeline Kurulumu
    # İlk çalıştırmada veriyi yükle ve RAG zincirini oturum durumuna kaydet
    if "qa_chain" not in st.session_state:
        documents = load_and_transform_data(CSV_FILE)
        if documents:
            db = setup_rag_pipeline(documents)
            st.session_state['qa_chain'] = create_qa_chain(db)
            st.session_state['messages'] = [{"role": "assistant", "content": "Deprem verileri yüklendi. Hangi deprem hakkında bilgi almak istersiniz?"}]
        else:
            return # Veri yoksa uygulamayı sonlandır

    # 4.3. Chat Geçmişinin Gösterilmesi
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4.4. Kullanıcıdan Girdi Alma
    if prompt := st.chat_input("Sorunuzu buraya yazın... (Örn: 6 Şubat 2023 depremi ile ilgili bilgi ver)"):
        
        # Kullanıcı mesajını geçmişe ekle
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4.5. RAG Zincirini Çalıştırma
        with st.spinner("Bilgi aranıyor ve yanıt oluşturuluyor..."):
            qa_chain = st.session_state['qa_chain']
            
            # Zinciri çalıştırma
            response = qa_chain({"query": prompt})

            # RAG Yanıtını İşleme
            answer = response["result"]
            
            # Kaynak Belgesini Ekleme (Proje kriteri için önemlidir)
            source_docs = response["source_documents"]
            sources = "\n".join([doc.page_content for doc in source_docs])
            
            # LLM'e cevabı kaynaklarla birlikte özetlemesi için System Prompt mantığı eklenebilir.
            final_response = f"{answer}\n\n**Kaynaklar (Retrieved Chunks):**\n```\n{sources}\n```"
        
        # 4.6. Asistan Cevabını Gösterme
        with st.chat_message("assistant"):
            st.markdown(final_response)
        
        # Cevabı geçmişe kaydet
        st.session_state['messages'].append({"role": "assistant", "content": final_response})


if __name__ == '__main__':
    main()
# ==============================================================================
# BÖLÜM 5: KOD ANLATIMININ BİTİŞİ
# Tüm teknik anlatımlar bu dosya içerisinde yorum satırları olarak yer almıştır.
# ==============================================================================
