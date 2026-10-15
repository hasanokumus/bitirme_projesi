Bu klasörde proje süresince kullanılmış çeşitli yapay zeka model eğitimi kodları vardır.

-**son_effnetb3.ipynb** ile eğitilen model EfficentNet B3 modeli ile STM32 üzerinde kullanılamayacak ancak yüksek hesaplama imkanlarında veri setinin ulaşabileceği maksimum başarıyı ölçmek için kullanılmıştır.
-**son_squeezenet.ipynb** ile eğitilen model kart üzerinde kullanılabilecek örnek teşkil etmesi için kullanılmıştır. 
-**son_stfd_mobilenetv1.ipynb** ile eğitilen nihai STFD_MobileNet V1 modeli ise kart üzerinde kullanılan ve %90.81 doğruluğa ulaşan modeldir.

Ayrıca **resim_ayirici.py** kodları da CVAT üzerinden alınan etiketlenmiş .yolo yapısındaki çıktılara göre resimlerden gerekli görüntüleri kırparak sınıflara ayırmak için kullanılmıştır.
