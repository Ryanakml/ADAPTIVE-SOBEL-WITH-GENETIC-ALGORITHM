import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import io
from mpl_toolkits.mplot3d import Axes3D
import random


# Konfigurasi halaman
st.set_page_config(
    page_title="Visualisasi Sobel Edge Detection",
    layout="wide"
)

# Judul aplikasi
st.title("Visualisasi Proses Sobel Edge Detection")
st.markdown("Aplikasi ini memvisualisasikan proses deteksi tepi menggunakan operator Sobel secara bertahap.")

# Fungsi untuk menerapkan Sobel edge detection
def apply_sobel(image):
    # Pastikan gambar dalam grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Terapkan operator Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Hitung magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalisasi untuk visualisasi
    sobel_x_norm = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_y_norm = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return gray, sobel_x, sobel_y, magnitude, sobel_x_norm, sobel_y_norm, magnitude_norm

# Fungsi untuk menerapkan thresholding
def apply_threshold(image, threshold_value):
    # Pastikan gambar sudah dinormalisasi
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Terapkan thresholding
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

# Fungsi untuk menerapkan adaptive thresholding
def apply_adaptive_threshold(image, method='gaussian', block_size=11, c=2):
    # Pastikan gambar sudah dinormalisasi
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Pastikan block_size ganjil
    if block_size % 2 == 0:
        block_size += 1
    
    # Pilih metode adaptive thresholding
    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:  # mean
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    
    # Terapkan adaptive thresholding
    binary = cv2.adaptiveThreshold(
        image, 
        255, 
        adaptive_method,
        cv2.THRESH_BINARY, 
        block_size, 
        c
    )
    return binary

# Fungsi untuk membuat quiver plot
def create_quiver_plot(sobel_x, sobel_y, step=8):
    # Buat grid untuk quiver plot
    h, w = sobel_x.shape
    y, x = np.mgrid[0:h:step, 0:w:step]
    
    # Subsample gradien untuk quiver plot
    fx = sobel_x[::step, ::step]
    fy = sobel_y[::step, ::step]
    
    # Normalisasi untuk visualisasi yang lebih baik
    magnitude = np.sqrt(fx**2 + fy**2)
    max_mag = np.max(magnitude)
    if max_mag > 0:  # Hindari pembagian dengan nol
        fx = fx / max_mag
        fy = fy / max_mag
    
    # Buat figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot heatmap magnitude
    im = ax.imshow(np.sqrt(sobel_x**2 + sobel_y**2), cmap='viridis')
    
    # Plot quiver (arah gradien)
    ax.quiver(x, y, fx, fy, color='white', scale=1, scale_units='xy')
    
    ax.set_title('Magnitude dan Arah Gradien')
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    # Hilangkan axis
    ax.axis('off')
    
    # Konversi plot ke image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Fungsi untuk membuat visualisasi 3D
def create_3d_visualization(image, title):
    # Buat figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Buat grid
    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    # Plot permukaan 3D
    surf = ax.plot_surface(x, y, image, cmap='viridis', linewidth=0, antialiased=False)
    
    # Tambahkan colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Atur judul dan label
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensitas')
    
    # Atur sudut pandang
    ax.view_init(30, 45)
    
    # Konversi plot ke image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Implementasi Algoritma Genetika untuk thresholding otomatis
class GeneticAlgorithm:
    def __init__(self, image, population_size=10, generations=5, mutation_rate=0.1):
        self.image = image
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_threshold = 127  # Nilai default
        self.best_fitness = 0
        self.history = []
        
    def initialize_population(self):
        # Inisialisasi populasi dengan nilai threshold acak
        return [random.randint(0, 255) for _ in range(self.population_size)]
    
    def fitness(self, threshold):
        # Fungsi fitness: mengevaluasi kualitas threshold
        # Menggunakan metode Otsu-like untuk memaksimalkan varians antar kelas
        _, binary = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        
        # Hitung histogram
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Hitung probabilitas kelas
        w0 = np.sum(hist[:threshold+1])
        w1 = 1 - w0
        
        if w0 == 0 or w1 == 0:
            return 0
        
        # Hitung mean kelas
        mu0 = np.sum(np.arange(0, threshold+1) * hist[:threshold+1]) / w0 if w0 > 0 else 0
        mu1 = np.sum(np.arange(threshold+1, 256) * hist[threshold+1:]) / w1 if w1 > 0 else 0
        
        # Hitung varians antar kelas
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        return variance
    
    def selection(self, population, fitnesses):
        # Seleksi berdasarkan fitness (roulette wheel)
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.sample(population, 2)
        
        selection_probs = [f/total_fitness for f in fitnesses]
        selected_indices = np.random.choice(len(population), 2, p=selection_probs)
        return [population[i] for i in selected_indices]
    
    def crossover(self, parent1, parent2):
        # Crossover sederhana: rata-rata nilai
        return int((parent1 + parent2) / 2)
    
    def mutation(self, individual):
        # Mutasi: tambah atau kurang nilai acak
        if random.random() < self.mutation_rate:
            change = random.randint(-20, 20)
            individual = max(0, min(255, individual + change))
        return individual
    
    def evolve(self):
        # Proses evolusi
        population = self.initialize_population()
        generation_data = []
        
        for generation in range(self.generations):
            # Evaluasi fitness
            fitnesses = [self.fitness(individual) for individual in population]
            
            # Temukan individu terbaik
            best_idx = np.argmax(fitnesses)
            current_best = population[best_idx]
            current_best_fitness = fitnesses[best_idx]
            
            # Simpan data generasi
            generation_data.append({
                'generation': generation,
                'population': population.copy(),
                'fitnesses': fitnesses.copy(),
                'best_threshold': current_best,
                'best_fitness': current_best_fitness
            })
            
            # Update best overall
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_threshold = current_best
            
            # Buat populasi baru
            new_population = []
            
            # Elitisme: simpan individu terbaik
            new_population.append(current_best)
            
            # Buat sisa populasi
            while len(new_population) < self.population_size:
                # Seleksi
                parents = self.selection(population, fitnesses)
                
                # Crossover
                child = self.crossover(parents[0], parents[1])
                
                # Mutasi
                child = self.mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        self.history = generation_data
        return self.best_threshold, self.history

# Fungsi untuk membuat visualisasi proses GA
def visualize_ga_process(ga_history):
    # Buat figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Evolusi nilai threshold terbaik
    generations = [data['generation'] for data in ga_history]
    best_thresholds = [data['best_threshold'] for data in ga_history]
    
    ax1.plot(generations, best_thresholds, 'o-', color='blue')
    ax1.set_title('Evolusi Nilai Threshold Terbaik')
    ax1.set_xlabel('Generasi')
    ax1.set_ylabel('Nilai Threshold')
    ax1.grid(True)
    
    # Plot 2: Distribusi populasi pada generasi terakhir
    last_gen = ga_history[-1]
    population = last_gen['population']
    fitnesses = last_gen['fitnesses']
    
    # Normalisasi fitness untuk ukuran marker
    if max(fitnesses) > 0:
        normalized_fitnesses = [50 * f / max(fitnesses) for f in fitnesses]
    else:
        normalized_fitnesses = [10 for _ in fitnesses]
    
    # Plot populasi
    ax2.scatter(population, [0] * len(population), s=normalized_fitnesses, alpha=0.7)
    ax2.axvline(x=last_gen['best_threshold'], color='red', linestyle='--', 
                label=f'Threshold Terbaik: {last_gen["best_threshold"]}')
    
    ax2.set_title('Distribusi Populasi (Generasi Terakhir)')
    ax2.set_xlabel('Nilai Threshold')
    ax2.set_yticks([])
    ax2.set_xlim(0, 255)
    ax2.legend()
    
    plt.tight_layout()
    
    # Konversi plot ke image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Terapkan Sobel
    gray, sobel_x, sobel_y, magnitude, sobel_x_norm, sobel_y_norm, magnitude_norm = apply_sobel(image)
    
    # Buat quiver plot
    quiver_buf = create_quiver_plot(sobel_x, sobel_y)
    quiver_img = Image.open(quiver_buf)
    
    # Buat visualisasi 3D
    gray_3d_buf = create_3d_visualization(gray, 'Visualisasi 3D Gambar Grayscale')
    sobel_x_3d_buf = create_3d_visualization(sobel_x, 'Visualisasi 3D Sobel X (Gx)')
    sobel_y_3d_buf = create_3d_visualization(sobel_y, 'Visualisasi 3D Sobel Y (Gy)')
    magnitude_3d_buf = create_3d_visualization(magnitude, 'Visualisasi 3D Magnitude')
    
    gray_3d_img = Image.open(gray_3d_buf)
    sobel_x_3d_img = Image.open(sobel_x_3d_buf)
    sobel_y_3d_img = Image.open(sobel_y_3d_buf)
    magnitude_3d_img = Image.open(magnitude_3d_buf)
    
    # Tampilkan visualisasi dengan expander (dropdown)
    with st.expander("Tahap 1: Gambar Input (Grayscale)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="Gambar Grayscale", use_container_width=True)
        with col2:
            st.image(gray_3d_img, caption="Visualisasi 3D Grayscale", use_container_width=True)
        
        st.markdown("""
        **Penjelasan:**
        
        Gambar input dikonversi ke grayscale (skala abu-abu) untuk memudahkan proses deteksi tepi.
        Operator Sobel bekerja dengan menganalisis perubahan intensitas piksel dalam gambar grayscale.
        Setiap piksel memiliki nilai intensitas antara 0 (hitam) hingga 255 (putih).
        
        **Visualisasi 3D** menunjukkan intensitas piksel sebagai ketinggian, sehingga area terang muncul sebagai "puncak" dan area gelap sebagai "lembah".
        """)
    
    with st.expander("Tahap 2: Hasil Sobel X (Gx)"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(sobel_x_norm, caption="Sobel X (Gx)", use_container_width=True)
        with col2:
            st.image(sobel_x_3d_img, caption="Visualisasi 3D Sobel X", use_container_width=True)
        
        st.markdown("""
        **Penjelasan:**
        
        Sobel X (Gx) mendeteksi tepi vertikal dengan mengukur perubahan intensitas piksel secara horizontal.
        Area terang menunjukkan perubahan intensitas yang kuat dari kiri ke kanan.
        Area gelap menunjukkan perubahan intensitas yang kuat dari kanan ke kiri.
        Area abu-abu (nilai mendekati nol) menunjukkan tidak ada perubahan intensitas horizontal yang signifikan.
        
        **Visualisasi 3D** menunjukkan gradien horizontal sebagai permukaan 3D, di mana puncak dan lembah menunjukkan perubahan intensitas yang kuat.
        """)
    
    with st.expander("Tahap 3: Hasil Sobel Y (Gy)"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(sobel_y_norm, caption="Sobel Y (Gy)", use_container_width=True)
        with col2:
            st.image(sobel_y_3d_img, caption="Visualisasi 3D Sobel Y", use_container_width=True)
        
        st.markdown("""
        **Penjelasan:**
        
        Sobel Y (Gy) mendeteksi tepi horizontal dengan mengukur perubahan intensitas piksel secara vertikal.
        Area terang menunjukkan perubahan intensitas yang kuat dari atas ke bawah.
        Area gelap menunjukkan perubahan intensitas yang kuat dari bawah ke atas.
        Area abu-abu (nilai mendekati nol) menunjukkan tidak ada perubahan intensitas vertikal yang signifikan.
        
        **Visualisasi 3D** menunjukkan gradien vertikal sebagai permukaan 3D, di mana puncak dan lembah menunjukkan perubahan intensitas yang kuat.
        """)
    
    with st.expander("Tahap 4: Magnitude dari Gx dan Gy"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(magnitude_norm, caption="Magnitude", use_container_width=True)
        with col2:
            st.image(magnitude_3d_img, caption="Visualisasi 3D Magnitude", use_container_width=True)
        
        st.markdown("""
        **Penjelasan:**
        
        Magnitude dihitung dari Gx dan Gy menggunakan rumus: Magnitude = √(Gx² + Gy²)
        
        **Heatmap Magnitude:**
        - Menunjukkan kekuatan tepi pada setiap piksel
        - Area terang menunjukkan tepi yang kuat
        - Area gelap menunjukkan tidak ada tepi
        
        **Visualisasi 3D Magnitude:**
        - Menunjukkan kekuatan tepi sebagai ketinggian dalam ruang 3D
        - Puncak tinggi menunjukkan tepi yang kuat
        - Area datar menunjukkan tidak ada tepi
        """)
    
    # with st.expander("Tahap 5: Quiver Plot (Arah Gradien)"):
    #     st.image(quiver_img, caption="Quiver Plot (Magnitude + Arah)", use_container_width=True)
        
    #     st.markdown("""
    #     **Penjelasan:**
        
    #     **Quiver Plot:**
    #     - Menggabungkan informasi magnitude (warna) dan arah gradien (panah)
    #     - Panah menunjukkan arah perubahan intensitas
    #     - Panjang panah menunjukkan kekuatan gradien
    #     - Warna latar belakang menunjukkan magnitude
        
    #     Arah panah menunjukkan arah tegak lurus terhadap tepi, yang merupakan arah gradien intensitas maksimum.
    #     """)
    
    # Fungsi untuk membuat histogram dengan visualisasi threshold
    def create_threshold_histogram(image, threshold_value):
        # Buat figure
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Hitung histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Plot histogram
        ax.bar(range(256), hist, width=1, color='gray', alpha=0.7)
        
        # Plot garis threshold
        ax.axvline(x=threshold_value, color='red', linestyle='--', 
                   label=f'Threshold: {threshold_value}')
        
        # Beri warna area di atas dan di bawah threshold
        ax.fill_between(range(0, threshold_value+1), hist[:threshold_value+1], 
                       alpha=0.3, color='blue', label='Piksel Hitam (0)')
        ax.fill_between(range(threshold_value, 256), hist[threshold_value:], 
                       alpha=0.3, color='yellow', label='Piksel Putih (255)')
        
        # Atur judul dan label
        ax.set_title('Histogram Intensitas Piksel')
        ax.set_xlabel('Intensitas Piksel')
        ax.set_ylabel('Jumlah Piksel')
        ax.legend()
        
        # Konversi plot ke image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf
    
    # Tambahkan bagian thresholding manual
    with st.expander("Tahap 6: Thresholding Manual"):
        st.markdown("""
        **Penjelasan:**
        
        Thresholding adalah proses mengubah gambar grayscale menjadi gambar biner (hitam dan putih) berdasarkan nilai ambang batas (threshold).
        Piksel dengan nilai di atas threshold akan menjadi putih (255), sedangkan piksel dengan nilai di bawah threshold akan menjadi hitam (0).
        
        Histogram di bawah menunjukkan distribusi intensitas piksel dalam gambar. Area biru menunjukkan piksel yang akan menjadi hitam, 
        dan area kuning menunjukkan piksel yang akan menjadi putih dengan threshold yang dipilih.
        
        Geser slider untuk melihat efek nilai threshold yang berbeda pada hasil deteksi tepi.
        """)
        
        threshold_value = st.slider("Nilai Threshold", 0, 255, 127, 1)
        
        # Buat histogram dengan visualisasi threshold
        hist_buf = create_threshold_histogram(magnitude_norm, threshold_value)
        hist_img = Image.open(hist_buf)
        
        # Tampilkan histogram
        st.image(hist_img, caption="Histogram Intensitas Piksel dengan Threshold", use_container_width=True)
        
        # Terapkan thresholding pada magnitude
        binary_edge = apply_threshold(magnitude_norm, threshold_value)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(magnitude_norm, caption="Magnitude Original", use_container_width=True)
        with col2:
            st.image(binary_edge, caption=f"Hasil Thresholding (Nilai: {threshold_value})", use_container_width=True)
    
    # Tambahkan bagian thresholding otomatis dengan GA
    with st.expander("Tahap 7: Thresholding Otomatis dengan Algoritma Genetika"):
        st.markdown("""
        **Penjelasan:**
        
        Algoritma Genetika (GA) adalah metode optimasi yang terinspirasi dari proses evolusi alami.
        Dalam konteks thresholding, GA mencari nilai threshold optimal yang memaksimalkan kualitas deteksi tepi.
        
        Proses GA melibatkan:
        1. Inisialisasi populasi (nilai threshold acak)
        2. Evaluasi fitness (kualitas threshold)
        3. Seleksi individu terbaik
        4. Crossover (perkawinan) dan mutasi
        5. Iterasi hingga menemukan solusi optimal
        
        Klik tombol di bawah untuk menjalankan GA dan menemukan threshold optimal.
        """)
        
        if st.button("Jalankan Algoritma Genetika"):
            with st.spinner("Menjalankan Algoritma Genetika..."):
                # Inisialisasi dan jalankan GA
                ga = GeneticAlgorithm(
                    magnitude_norm, 
                    population_size=20, 
                    generations=10, 
                    mutation_rate=0.2
                )
                optimal_threshold, ga_history = ga.evolve()
                
                # Terapkan threshold optimal
                binary_edge_ga = apply_threshold(magnitude_norm, optimal_threshold)
                
                # Visualisasi proses GA
                ga_viz_buf = visualize_ga_process(ga_history)
                ga_viz_img = Image.open(ga_viz_buf)
                
                # Tampilkan hasil
                st.success(f"Threshold optimal yang ditemukan: {optimal_threshold}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(binary_edge_ga, caption=f"Hasil Thresholding Otomatis (Nilai: {optimal_threshold})", use_container_width=True)
                with col2:
                    st.image(ga_viz_img, caption="Visualisasi Proses Algoritma Genetika", use_container_width=True)
                
                # Tampilkan detail evolusi
                st.subheader("Detail Evolusi GA")
                
                # Buat tabel evolusi
                evolution_data = []
                for gen in ga_history:
                    evolution_data.append({
                        "Generasi": gen['generation'] + 1,
                        "Threshold Terbaik": gen['best_threshold'],
                        "Fitness Terbaik": f"{gen['best_fitness']:.6f}"
                    })
                
                st.table(evolution_data)

    # Fungsi untuk memvisualisasikan adaptive thresholding
    def visualize_adaptive_threshold(image, block_size, c, method='gaussian'):
        # Pastikan gambar sudah dinormalisasi
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Buat figure dengan 2x2 subplot
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Gambar asli
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 0].set_title('Gambar Original')
        axs[0, 0].axis('off')
        
        # 2. Visualisasi threshold lokal
        # Buat heatmap yang menunjukkan nilai threshold lokal
        local_threshold_map = np.zeros_like(image, dtype=np.float32)
        
        # Hitung threshold lokal untuk setiap piksel
        half_block = block_size // 2
        for i in range(half_block, image.shape[0] - half_block):
            for j in range(half_block, image.shape[1] - half_block):
                # Ambil area lokal
                local_region = image[i-half_block:i+half_block+1, j-half_block:j+half_block+1]
                
                # Hitung threshold lokal berdasarkan metode
                if method == 'gaussian':
                    # Gaussian weighted mean
                    kernel = cv2.getGaussianKernel(block_size, 0)
                    kernel = kernel @ kernel.T
                    weighted_sum = np.sum(local_region * kernel)
                    weighted_count = np.sum(kernel)
                    local_threshold = weighted_sum / weighted_count - c
                else:
                    # Simple mean
                    local_threshold = np.mean(local_region) - c
                
                local_threshold_map[i, j] = local_threshold
        
        # Normalisasi untuk visualisasi
        local_threshold_map = cv2.normalize(local_threshold_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Plot heatmap threshold lokal
        im = axs[0, 1].imshow(local_threshold_map, cmap='viridis')
        axs[0, 1].set_title(f'Heatmap Threshold Lokal ({method})')
        axs[0, 1].axis('off')
        plt.colorbar(im, ax=axs[0, 1], label='Nilai Threshold')
        
        # 3. Hasil adaptive thresholding
        binary_adaptive = apply_adaptive_threshold(image, method, block_size, c)
        axs[1, 0].imshow(binary_adaptive, cmap='gray')
        axs[1, 0].set_title(f'Hasil Adaptive Thresholding\nBlock Size: {block_size}, C: {c}')
        axs[1, 0].axis('off')
        
        # 4. Perbandingan dengan global thresholding
        # Gunakan metode Otsu untuk mendapatkan threshold global optimal
        otsu_thresh, binary_global = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        axs[1, 1].imshow(binary_global, cmap='gray')
        axs[1, 1].set_title(f'Global Thresholding (Otsu)\nThreshold: {int(otsu_thresh)}')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Konversi plot ke image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf, local_threshold_map, binary_adaptive, binary_global, otsu_thresh

    # Fungsi untuk membuat visualisasi perbandingan detail
    def visualize_adaptive_vs_global_detail(image, binary_adaptive, binary_global, block_size):
        # Pilih area kecil untuk perbandingan detail
        h, w = image.shape
        # Pilih area di tengah gambar
        center_y, center_x = h // 2, w // 2
        size = min(100, h // 4, w // 4)  # Ukuran area detail
        
        # Tentukan koordinat area detail
        y1, y2 = center_y - size, center_y + size
        x1, x2 = center_x - size, center_x + size
        
        # Potong area detail
        detail_orig = image[y1:y2, x1:x2]
        detail_adaptive = binary_adaptive[y1:y2, x1:x2]
        detail_global = binary_global[y1:y2, x1:x2]
        
        # Buat figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot area detail
        axs[0].imshow(detail_orig, cmap='gray')
        axs[0].set_title('Detail Gambar Original')
        axs[0].axis('off')
        
        axs[1].imshow(detail_adaptive, cmap='gray')
        axs[1].set_title(f'Detail Adaptive Thresholding\nBlock Size: {block_size}')
        axs[1].axis('off')
        
        axs[2].imshow(detail_global, cmap='gray')
        axs[2].set_title('Detail Global Thresholding')
        axs[2].axis('off')
        
        # Tambahkan grid untuk menunjukkan ukuran block
        for ax in axs:
            # Tambahkan kotak yang menunjukkan ukuran block
            rect = plt.Rectangle((detail_orig.shape[1]//2 - block_size//2, 
                                detail_orig.shape[0]//2 - block_size//2), 
                                block_size, block_size, 
                                linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.tight_layout()
        
        # Konversi plot ke image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf
        
    # Fungsi untuk membuat visualisasi pengaruh parameter
    def visualize_parameter_effect(image):
        # Buat figure dengan 3x3 subplot
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        
        # Variasi block size
        block_sizes = [3, 11, 25]
        c_values = [-5, 2, 10]
        
        # Terapkan adaptive thresholding dengan berbagai parameter
        for i, block_size in enumerate(block_sizes):
            for j, c in enumerate(c_values):
                binary = apply_adaptive_threshold(image, 'gaussian', block_size, c)
                axs[i, j].imshow(binary, cmap='gray')
                axs[i, j].set_title(f'Block: {block_size}, C: {c}')
                axs[i, j].axis('off')
        
        plt.tight_layout()
        
        # Konversi plot ke image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf

    # # Tambahkan bagian adaptive thresholding
    # with st.expander("Tahap 8: Adaptive Thresholding"):
    #     st.markdown("""
    #     **Penjelasan:**
        
    #     Adaptive Thresholding adalah metode thresholding di mana nilai threshold dihitung secara lokal untuk setiap piksel berdasarkan piksel-piksel tetangganya.
        
    #     Berbeda dengan thresholding global (manual atau GA) yang menggunakan satu nilai threshold untuk seluruh gambar, adaptive thresholding menyesuaikan threshold berdasarkan kondisi lokal, sehingga lebih efektif untuk gambar dengan pencahayaan tidak merata.
        
    #     Ada dua metode utama:
    #     1. **Mean**: Threshold adalah rata-rata dari piksel tetangga
    #     2. **Gaussian**: Threshold adalah rata-rata tertimbang Gaussian dari piksel tetangga
        
    #     Parameter penting:
    #     - **Block Size**: Ukuran area tetangga yang digunakan untuk menghitung threshold
    #     - **C**: Konstanta yang dikurangkan dari rata-rata atau rata-rata tertimbang
    #     """)
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         method = st.radio("Metode Adaptive Thresholding", ["Gaussian", "Mean"])
    #         block_size = st.slider("Block Size (harus ganjil)", 3, 51, 11, 2)
    #         c_value = st.slider("Nilai C", -10, 30, 2)
        
    #     # Terapkan adaptive thresholding dan visualisasi
    #     adaptive_method = 'gaussian' if method == 'Gaussian' else 'mean'
        
    #     # Visualisasi adaptive thresholding
    #     viz_buf, local_threshold_map, binary_adaptive, binary_global, otsu_thresh = visualize_adaptive_threshold(
    #         magnitude_norm, block_size, c_value, adaptive_method
    #     )
    #     viz_img = Image.open(viz_buf)
        
    #     with col2:
    #         st.image(binary_adaptive, caption=f"Hasil Adaptive Thresholding ({method})", use_container_width=True)
        
    #     # Tampilkan visualisasi utama
    #     st.image(viz_img, caption="Visualisasi Adaptive Thresholding", use_container_width=True)
        
    #     # Visualisasi detail perbandingan
    #     st.subheader("Perbandingan Detail: Adaptive vs Global Thresholding")
    #     st.markdown("""
    #     Visualisasi di bawah menunjukkan perbandingan detail antara adaptive thresholding dan global thresholding.
    #     Kotak merah menunjukkan ukuran block yang digunakan untuk menghitung threshold lokal.
    #     """)
        
    #     detail_buf = visualize_adaptive_vs_global_detail(magnitude_norm, binary_adaptive, binary_global, block_size)
    #     detail_img = Image.open(detail_buf)
    #     st.image(detail_img, caption="Perbandingan Detail", use_container_width=True)
        
    #     # Visualisasi pengaruh parameter
    #     st.subheader("Pengaruh Parameter pada Adaptive Thresholding")
    #     st.markdown("""
    #     Visualisasi di bawah menunjukkan bagaimana perubahan parameter Block Size dan C mempengaruhi hasil adaptive thresholding.
    #     - **Block Size yang lebih besar**: Menangkap variasi intensitas dalam area yang lebih luas
    #     - **Block Size yang lebih kecil**: Lebih sensitif terhadap detail lokal
    #     - **C yang lebih besar**: Menghasilkan lebih banyak piksel putih (tepi lebih tebal)
    #     - **C yang lebih kecil**: Menghasilkan lebih sedikit piksel putih (tepi lebih tipis)
    #     """)
        
    #     param_buf = visualize_parameter_effect(magnitude_norm)
    #     param_img = Image.open(param_buf)
    #     st.image(param_img, caption="Pengaruh Parameter pada Adaptive Thresholding", use_container_width=True)
        
    #     st.markdown("""
    #     **Kelebihan Adaptive Thresholding:**
        
    #     1. Bekerja lebih baik pada gambar dengan pencahayaan tidak merata
    #     2. Dapat mendeteksi tepi halus yang mungkin hilang dengan thresholding global
    #     3. Tidak memerlukan pencarian nilai threshold optimal secara manual atau otomatis
        
    #     **Kekurangan Adaptive Thresholding:**
        
    #     1. Lebih lambat dibandingkan thresholding global
    #     2. Sensitif terhadap noise pada gambar
    #     3. Memerlukan penyesuaian parameter (block size dan C) untuk hasil optimal
        
    #     **Cara Kerja Adaptive Thresholding:**
        
    #     1. Untuk setiap piksel, ambil area tetangga dengan ukuran block_size × block_size
    #     2. Hitung threshold lokal berdasarkan metode yang dipilih:
    #        - Mean: Rata-rata intensitas piksel tetangga
    #        - Gaussian: Rata-rata tertimbang Gaussian dari piksel tetangga
    #     3. Kurangkan nilai C dari threshold lokal
    #     4. Jika piksel > threshold lokal, maka piksel = 255 (putih), jika tidak maka piksel = 0 (hitam)
        
    #     Heatmap threshold lokal menunjukkan bagaimana nilai threshold bervariasi di seluruh gambar, menyesuaikan dengan kondisi lokal.
    #     """)

else:
    # Tampilkan gambar contoh jika tidak ada yang diunggah
    st.info("Silakan unggah gambar untuk memulai visualisasi proses Sobel edge detection.")
    
    # Buat gambar contoh sederhana
    example_img = np.zeros((300, 300), dtype=np.uint8)
    # Buat bentuk kotak di tengah
    example_img[100:200, 100:200] = 255
    
    st.image(example_img, caption="Contoh: Unggah gambar untuk melihat hasil yang sebenarnya", width=300)




