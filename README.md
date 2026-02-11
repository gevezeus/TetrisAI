# 🧠 Tetris AI - Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning)

Bu proje, **Deep Q-Learning (DQN)** kullanarak kendi kendine Tetris oynamayı öğrenen bir yapay zeka uygulamasıdır.

Yapay zeka sadece ekrandaki pikselleri görmekle kalmaz; tahtanın durumunu (boşluklar, yüzey pürüzlülüğü, yükseklik) analiz eder ve hatta **bir sonraki parçayı da düşünerek** (2-Step Lookahead) en iyi hamleyi hesaplar.

## 🚀 Özellikler

*   **Deep Q-Network (DQN):** En iyi hamleleri tahmin etmek için bir Sinir Ağı (PyTorch) kullanır.
*   **Ödül Sistemi (Reward Shaping):** Satır silmeyi ödüllendirir; boşluk bırakmayı, yüzeyi bozmayı ve yükselmeyi cezalandırır.
*   **2-Adım İleri Görüş (2-Step Lookahead):** Yapay zeka sadece elindeki parçayı değil, bir sonraki parçanın nereye oturacağını da hesaplayarak oynar.
*   **Görselleştirme:** Yapay zekanın hamlelerini animasyonlu şekilde izleyebilirsiniz.
*   **Canlı Kontrol:** Oyun hızını klavye ile anlık olarak değiştirebilirsiniz.

## 🛠 Kurulum

1.  **Depoyu (Repository) indirin:**
    ```bash
    git clone https://github.com/kullaniciadiniz/TetrisAI.git
    cd TetrisAI
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    Sanal ortam (virtual environment) kullanmanız önerilir.
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Kullanım

Eğitimi ve oyunu başlatmak için ana dosyayı çalıştırın:

```bash
python main.py
```

### ⌨️ Kontroller (Oyun Sırasında)

*   **W:** Hızlandır (Zamanı ileri sar)
*   **S:** Yavaşlat (Hamleleri incele)
*   **Q:** Çıkış
*   **Ctrl+C:** Eğitimi Durdur (İlerleme otomatik kaydedilir)

## 🧠 Nasıl Çalışır?

### Beyin (Yapay Sinir Ağı)
Yapay zeka, aşağıdaki girdileri alan 512 nöronlu bir Tam Bağlantılı Sinir Ağı (Fully Connected Neural Network) kullanır:
1.  **Silinen Satır:** Bu hamle kaç satır silecek? (Büyük Ödül!)
2.  **Boşluklar:** Altta gömülü boşluk kalıyor mu? (Ceza!)
3.  **Pürüzlülük:** Yüzey düz mü yoksa engebeli mi? (Ceza!)
4.  **Toplam Yükseklik:** Kule çok mu yükseldi? (Ceza!)

### Eğitim Süreci
1.  **Keşif (Exploration):** Başlangıçta yapay zeka rastgele hamleler yaparak (`epsilon=1.0`) oyun kurallarını keşfeder.
2.  **Öğrenme (Learning):** Yaptığı her hamleyi ve sonucunu hafızasına kaydeder.
3.  **Optimizasyon:** Geçmiş tecrübelerinden rastgele örnekler alarak hatalarını azaltacak şekilde kendini günceller.
4.  **Uygulama (Exploitation):** Zamanla rastgeleliği azaltır ve öğrendiği stratejileri uygulamaya başlar (`epsilon -> 0.01`).

### Kaydetme & Yükleme
Model, her 25 bölümde bir ilerlemesini `tetris_dqn.pth` dosyasına otomatik olarak kaydeder. Programı kapatıp açtığınızda kaldığı yerden (öğrendiği zeka seviyesinden) devam eder.

## 📊 Performans
*   **Bölüm 0-100:** Rastgele hareketler, çok nadir satır siler.
*   **Bölüm 500+:** Düz zeminler oluşturmaya ve boşluklardan kaçınmaya başlar.
*   **Bölüm 1000+:** "Tetris" hamleleri (aynı anda 4 satır silme) yapmaya başlar ve oyunu çok uzun süre sürdürebilir.

## 📝 Lisans
MIT Lisansı. İstediğiniz gibi kullanabilir ve geliştirebilirsiniz!
