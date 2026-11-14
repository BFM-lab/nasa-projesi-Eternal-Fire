# 🚀 Mars EVA Görev Kontrolü & Hayatta Kalma Kılavuzu (Eternal Fire)

> **NASA Space Apps Challenge**  
> **Kategori / Challenge:** Interplanetary Survival Guide: Martian Map  
> **Yerel Etkinlik:** Bursa  
> **Takım Adı:** Eternal Fire  
> **Ekip:** Hüseyin Uğur · Ahmet Buğra Girgin · Ege Kaan Akış · Oğuz Kağan Er · Berkay Yıldırım  

🔗 **Canlı Uygulama Linki:** [eternal-fire.streamlit.app](https://eternal-fire-86f98zyhvpmlwffjmxj7jh.streamlit.app/)

---

## 📌 Proje Genel Bakış

Mars yüzeyinde gerçekleştirilecek araç dışı aktivitelerde (EVA - Extravehicular Activity) en kısa geometrik rota, ölümcül bir tuzak olabilir. Dik krater duvarları, aşırı düşük gölge sıcaklıkları ve radyasyon fırtınaları astronotun uzay giysisi limitlerini dakikalar içinde tüketebilir.

**Mars EVA Görev Kontrolü**, Gale ve Jezero kraterlerindeki 3 km × 3 km'lik hiper-yerel keşif sahalarında iniş aracından (Lander) hedeflenen bilimsel numune sahasına gidiş ve dönüş için **A\*** tabanlı çok kriterli en güvenli rotayı çizen ve hayatta kalma telemetrisini hesaplayan interaktif bir taktik planlayıcıdır.

---

## 🛰️ Kullanılan Gerçek NASA Verileri ve Bilimsel Modeller

Proje, varsayımsal değerler yerine doğrudan NASA görevlerinden elde edilen veriler ve fizik yasaları üzerine inşa edilmiştir:

1. **Arazi & Topografya (Ham Veri):** NASA JPL Mars Trek platformundan alınan 3D DEM (Dijital Yükseklik Modeli) `.stl` yüzey ağları işlenerek 15 metrelik hücre çözünürlüğünde ($200 \times 200$ grid) yükseklik, eğim açısı ve krater derinlik katmanları çıkarılmıştır.
2. **Saatlik Termal Model:** 
   - **Gale Krateri:** Mars Science Laboratory (Curiosity) üzerindeki **REMS** (Rover Environmental Monitoring Station) cihazının 24 saatlik Sol sıcaklık eğrisi.
   - **Jezero Krateri:** Mars 2020 (Perseverance) üzerindeki **MEDA** (Mars Environmental Dynamics Analyzer) cihazının 24 saatlik Sol sıcaklık eğrisi.
   - Anlık Sol saatine göre güneş irtifa/azimut açısı hesaplanarak dinamik yüzey gölgelenmesi (`hillshade`) ve krater çukurlarında soğuk hava birikimi termal haritaya yansıtılır.
3. **Yüzey Radyasyonu:** Curiosity **RAD** (Radiation Assessment Detector) tarafından ölçülen Mars yüzey baz dozu (~0.64 mSv/gün) kullanılmış; güneş fırtınası senaryosunda doz 4 katına çıkarılarak arazi siperlenmesi (shielding) hesaba katılmıştır.
4. **Biyomekanik ve Yaşam Destek Harcaması:** Mars yerçekimi ($g = 3.72 \text{ m/s}^2$), giysi/ekipman kütlesi ($150 - 195 \text{ kg}$), eğim tırmanma potansiyel enerjisi ve sıcaklığa bağlı aktif giysi ısıtma gücü (Watt) üzerinden batarya ve oksijen tüketimi hesaplanır.

---

## ⚙️ Temel Özellikler

- **2D Taktik & 3D İnteraktif Arazi Görünümü:** Eğim, krater derinliği, saatlik yüzey sıcaklığı, radyasyon ve geçilebilirlik engellerinin Plotly destekli görselleştirilmesi.
- **A\* Tabanlı Hayatta Kalma Rotası:** Dik eğimleri, aşırı soğuk krater diplerini ve tehlikeli arazileri cezalandıran / bloke eden dinamik maliyet matrisi.
- **Kör En Kısa Rota (Naive) Kıyası:** Standart en kısa yol ile güvenlik optimizasyonlu rota arasındaki mesafe, batarya, eğim ve sıcaklık farkını tablo ve grafiklerle kanıtlama.
- **Dinamik Gidiş - Dönüş Planı:** Mars'ta saatler ilerledikçe sıcaklık düştüğü için gidiş rotası ile saatler sonraki dönüş rotası farklı termal koşullara göre bağımsız hesaplanır.
- **Adım Adım Taktik Yol Tarifi:** Pusula yönü, mesafe ve arazi tehlike uyarıları içeren Google Maps tarzı navigasyon adımları.
- **Çoklu Mobilite Profilleri:** Standart EVA, Ağır Bilimsel Ekipmanlı Keşif ve Rover Destekli Keşif modları.
- **Tıkanma Analizi:** Güvenli geçit bulunamadığında en yakın güvenli yaklaşma noktasını ve gereken asgari arazi toleransını sunan geri bildirim mekanizması.
- **İndirilebilir Brifing:** Operasyonel görev planının Markdown formatında saha kılavuzu olarak tek tıkla indirilmesi.

---

## 📁 Proje Dosya Yapısı

```text
├── raw/                 # NASA Mars Trek'ten indirilen Gale ve Jezero .stl modelleri
├── processed/           # data_prep.py tarafından üretilen npz ve sites.json arazi katmanları
├── images/              # Bilimsel hedeflere ait yüzey referans fotoğrafları
├── .streamlit/          # Streamlit yapılandırma dosyaları
├── data_prep.py         # STL geometrisini grid katmanlarına ve fiziksel haritalara dönüştüren script
├── survival_model.py    # Termal, radyasyon, enerji tüketimi ve maliyet ızgarası (cost grid) motoru
├── pathfinder.py        # A* arama algoritması, alternatif rota çözücü ve navigasyon adımları
├── app.py               # Streamlit web arayüzü ve 2D/3D Plotly görselleştirmesi
├── requirements.txt     # Python kütüphane bağımlılıkları
└── mars_eva_qr.png      # Uygulama canlı linkine giden karekod
🛠️ Yerel Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için:

    Repoyu klonlayın:
    

    git clone https://github.com/BFM-lab/nasa-projesi-Eternal-Fire.git
    cd nasa-projesi-Eternal-Fire

    Gereksinimleri yükleyin:
    

    pip install -r requirements.txt

    (Gerekirse) Veri katmanlarını derleyin:
    processed/ klasöründeki önceden işlenmiş verileri baştan üretmek isterseniz:
    

    python data_prep.py

    Uygulamayı başlatın:
    

    streamlit run app.py

🤖 Yapay Zeka Kullanım Beyanı (AI Disclosure)

NASA Space Apps Challenge şeffaflık kuralları doğrultusunda:
Projenin geliştirilme sürecinde veri ön işleme algoritmalarının optimizasyonu, dokümantasyon yapılandırması ve kod refactoring aşamalarında Google Gemini yapay zeka modelinden destek alınmıştır. Bilimsel modellerin mantığı, formülasyon parametreleri ve proje kurgusu takımımız tarafından doğrulanmış ve test edilmiştir.
📜 Lisans

Bu proje MIT Lisansı kapsamında açık kaynak olarak sunulmaktadır.

---

