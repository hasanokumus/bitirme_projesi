# STM32 Tabanlı Domates Yaprağı Sağlık Durumu Sınıflandırma

## Proje Hakkında

Bu proje, sera ortamında çekilmiş domates yaprağı görüntülerini kullanarak beş farklı sağlık durumunu sınıflandıran bir yapay zeka modelinin geliştirilmesini ve STM32F746G mikrodenetleyicisi üzerinde çalıştırılmasını kapsamaktadır.

## Projenin Aşamaları
- Sera ortamında çekilen görüntülerden veri seti oluşturulması ve etiketlenmesi.
- Farklı yapay zekâ modellerinin eğitilmesi ve sonuçlarının karşılaştırılması.
- Modelin nicemleme (quantization) ve gömülü sistemlere uygunluk açısından optimize edilmesi.
- Modelin STM32F746G kartına entegre edilmesi ve gerçek donanım üzerinde test edilmesi.

## Sonuçlar
- **Sınıflandırma doğruluğu:** %90,81
- **Test görüntüsü sayısı:** 1.405
- **Mikrodenetleyici:** STM32F746G
- **Dahili bellek kapasitesi:** 320 KB SRAM ve 1 MB Flash

## Depo İçeriği
- `ai-model/` — Yapay zekâ modelinin eğitim ve değerlendirme kodları.
- `stm32/` — STM32 üzerinde çıkarım ve gömülü yazılım kodları.
- `docs/` — Proje raporu ve dokümantasyon.
