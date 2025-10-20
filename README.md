# 🌍 RAG Tabanlı Deprem Bilgilendirme Chatbotu Projesi

## 1. Projenin Amacı

Bu projenin temel amacı, RAG (Retrieval Augmented Generation) mimarisi kullanarak **deprem verilerine dayalı** bir bilgilendirme chatbotu geliştirmektir. Chatbot, kullanıcıların belirli bir bölge veya tarih aralığındaki depremler hakkında sorduğu sorulara, elindeki bilgi tabanını kullanarak doğru ve bağlamsal yanıtlar verecektir. Proje, geliştirilen çözümü bir web arayüzü üzerinden sunmayı hedeflemektedir.

## 2. Veri Seti Hakkında Bilgi

* **Kaynak:** Projenin temel bilgi kaynağı, Kaggle'dan alınan "Turkey Earthquake Data 1914-2023" veri setidir.
* **Veri Tipi:** Veri seti orijinalinde deprem büyüklüğü (`Mw`), oluş tarihi, derinlik ve konum (`Yer`) gibi tablosal (CSV) verilerden oluşmaktadır.
* **RAG Dönüşümü:** RAG sisteminin metin tabanlı çalışma kısıtlaması nedeniyle, tablosal veriler doğrudan kullanılmamış, bunun yerine her bir deprem kaydı için bilgilendirici metinler türetilerek chatbotun bilgi tabanı (Knowledge Base) oluşturulmuştur.

## 3. Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, **Python** tabanlı RAG mimarisini kullanmaktadır.

| Bileşen | Kullanılan Teknoloji | Amaç |
| :--- | :--- | :--- |
| **RAG Framework** | LangChain | RAG pipeline'ı oluşturmak (Chunking, Retrieval, Generation adımlarını yönetmek). |
| **Generative Model (LLM)** | Gemini API | Sorgu bağlamına göre nihai ve anlamlı yanıtı üretmek. |
| **Embedding Model** | Google'ın veya açık kaynaklı bir model | Metin parçalarını (chunks) ve kullanıcı sorgularını sayısal vektörlere dönüştürmek. |
| **Vector Database** | ChromaDB | Deprem bilgilerinin vektörlerini saklamak ve hızlı anlamsal arama (Semantic Search) yapmak. |
| **Web Arayüzü** | Streamlit | Chatbot'u web üzerinden kullanıcıya sunmak. |

## 4. Elde Edilen Sonuçlar ve Proje Yetenekleri (Özet)

* **Bilgilendirme:** Kullanıcının spesifik deprem olayları (tarih, büyüklük, konum) hakkındaki sorularına, oluşturulan bilgi tabanına dayanarak yanıt verir.
* **Bağlamsal Yanıt:** RAG mimarisi sayesinde, LLM'in halüsinasyon yapma riski en aza indirilmiş ve yanıtlar yalnızca sağlanan verilere dayandırılmıştır.
* **Erişilebilirlik:** Chatbot, Streamlit kullanılarak geliştirilen kullanıcı dostu bir web arayüzü üzerinden erişilebilir durumdadır.

***

## 5. Projenin Çalışma Kılavuzu 

Bu kılavuz, projenin başarılı bir şekilde çalıştırılması için gereken adımları içermektedir.

**1. Gerekli Dosyaları İndirme:**

* GitHub deposunu yerel makinenize klonlayın:
    ```bash
    git clone [REPO LİNKİNİZ]
    cd [PROJE DİZİN ADI]
    ```
* Orijinal veri setini (Kaggle: `veriler.csv`) projenizin ana dizinine indirin.

**2. Sanal Ortam Kurulumu:**

* Python sanal ortamı oluşturun:
    ```bash
    python -m venv venv
    ```
* Sanal ortamı etkinleştirin:
    * **Windows:** `.\venv\Scripts\activate`
    * **Linux/macOS:** `source venv/bin/activate`

**3. Bağımlılıkların Yüklenmesi:**

* `requirements.txt` dosyasındaki tüm paketleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

**4. API Anahtarı Ayarı:**

* Proje dizininde bulunan `.env` dosyasını açın.
* Kendi Gemini API anahtarınızı `GEMINI_API_KEY=` satırına ekleyin.
    ```
    GEMINI_API_KEY=AIzaSy...
    ```

**5. Projenin Çalıştırılması:**

* Proje, Streamlit ile sunulmaktadır. Aşağıdaki komutla ana uygulamayı çalıştırın:
    ```bash
    streamlit run app.py  # app.py, ana python dosyanızın adı olmalıdır
    ```
* Uygulama, otomatik olarak tarayıcınızda açılacaktır.

*(Not: Proje kodunuzun tüm teknik anlatımlarına, Python dosyası içerisinde yorum satırları (`#`) ile yer verilmelidir.)*

***

## 6. Web Arayüzü & Product Kılavuzu

[Bu alana, proje tamamlandıktan sonra arayüzün ekran görüntüsü/video anlatımı ve test sorguları eklenecektir. Proje henüz tamamlanmadığı için şu an boş bırakılmıştır.]

**🌐 Canlı Uygulama Linki:** (Proje Dağıtımı Yapıldığında Buraya Eklenecektir)
