import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

print("PyCUDA başarıyla başlatıldı!")
print("Kullanılan Cihaz:", cuda.Device(0).name())

# 1. CPU'da (Host) rastgele verilerden oluşan bir matris oluştur
print("\n--- Veri Transferi Testi ---")
host_data = np.random.randn(1024, 1024).astype(np.float32)

# 2. GPU'da (Device) verinin boyutu kadar bellek ayır
device_data = cuda.mem_alloc(host_data.nbytes)

# 3. CPU'daki veriyi GPU'ya kopyala (Host to Device)
cuda.memcpy_htod(device_data, host_data)
print("1. Veri RAM'den VRAM'e (GPU) başarıyla kopyalandı.")

# 4. GPU'dan veriyi geri almak için CPU'da boş bir alan oluştur
host_data_returned = np.empty_like(host_data)

# 5. GPU'daki veriyi CPU'ya geri çek (Device to Host)
cuda.memcpy_dtoh(host_data_returned, device_data)
print("2. Veri VRAM'den RAM'e (CPU) başarıyla geri çekildi.")

# 6. Gönderilen ve alınan verileri karşılaştır
if np.allclose(host_data, host_data_returned):
    print("\nSONUÇ: BAŞARILI! Veriler bit-başına (bit-perfect) eşleşiyor.")
else:
    print("\nSONUÇ: HATA! Veri transferinde kayıp veya bozulma var.")