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

## 5. Projenin Çalışma Kılavuzu (Detaylar 1.2. Aşamada Verilecektir)

Projenin yerel veya bulut ortamında çalıştırılması için gerekli adımlar ve bağımlılıklar (Virtual Environment kurulumu, `requirements.txt` ile paket yükleme, API anahtarı ayarları) detaylı olarak açıklanacaktır.

***

## 6. Web Arayüzü & Product Kılavuzu

[Bu alana, proje tamamlandıktan sonra arayüzün ekran görüntüsü/video anlatımı ve test sorguları eklenecektir. Proje henüz tamamlanmadığı için şu an boş bırakılmıştır.]

**🌐 Canlı Uygulama Linki:** (Proje Dağıtımı Yapıldığında Buraya Eklenecektir)
