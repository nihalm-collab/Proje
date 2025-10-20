# Akbank GenAI Bootcamp Projesi: RAG Tabanlı Deprem Bilgilendirme Chatbotu

## [cite_start]🚀 Projenin Amacı [cite: 9]
[cite_start]Bu projenin temel amacı, Retrieval Augmented Generation (RAG) mimarisini kullanarak Türkiye'deki tarihsel deprem verilerine dayalı spesifik soruları yanıtlayabilen bir chatbot geliştirmektir[cite: 2]. [cite_start]Chatbot, web arayüzü üzerinden kullanıcıya sunulacaktır[cite: 2].

## [cite_start]📊 Veri Seti Hakkında Bilgi [cite: 10]
* **Adı:** Turkey Earthquake Data 1914-2023
* **Kaynak:** Kaggle
* **İçerik Özeti:** Veri seti, 1914-2023 yılları arasında Türkiye'de meydana gelen depremlere ait zaman, yer, büyüklük, derinlik gibi temel sismik bilgileri içermektedir. Bu veriler, chatbot'un bilgi kaynağını (Knowledge Base) oluşturacaktır.

## [cite_start]🛠️ Kullanılan Yöntemler [cite: 11]
Proje, RAG (Retrieval Augmented Generation) prensibine dayanmaktadır. Bu mimari şunları içerir:
1.  **Veri Ön İşleme ve Parçalama (Chunking):** Deprem verisinin okunması ve vektörleştirmeye uygun parçalara ayrılması.
2.  [cite_start]**Embedding:** Parçalanan metinlerin vektör uzayına dönüştürülmesi[cite: 43].
3.  [cite_start]**Vektör Veritabanı (Vector DB):** Vektörlerin depolanması ve kullanıcı sorgularına yanıt bulmak için hızlıca aranması[cite: 43].
4.  [cite_start]**Generation Model (LLM):** Kullanıcı sorusu ve ilgili veriler kullanılarak nihai yanıtın oluşturulması[cite: 42].

*Kullanılacak Teknolojiler (Geliştirme aşamasında kesinleşecektir):*
* [cite_start]**RAG Framework:** LangChain veya Haystack [cite: 44]
* [cite_start]**Generation Model:** Gemini API [cite: 42]
* [cite_start]**Vektör Database:** Chroma veya FAISS [cite: 43]

## [cite_start]💡 Elde Edilen Sonuçlar (Özet) [cite: 12]
*(Bu bölüm, proje tamamlandığında elde edilen başarımların özetlenmesi için şimdilik boş bırakılmıştır.)*

## [cite_start]🌐 Web Arayüzü Linki [cite: 13]
*(Bu link, chatbot deploy edildikten sonra buraya eklenecektir.)*
