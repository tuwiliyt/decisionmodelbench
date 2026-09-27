# Kekeliruan Autoregresi dalam Sistem Siber-Fisik Kritis-Waktu dan Perdagangan Algoritmik: Tolok Ukur Empiris Model Keputusan Non-Autoregresif Melawan Model Bahasa Besar Fondasional

**Richie O. Sumual**  
*Advanced Agentic AI & Distributed Systems Research*  
Jakarta, Indonesia  
`richie.sumual@research.org`

---

### Abstrak
Kecenderungan mutakhir untuk menerapkan Model Bahasa Besar (Large Language Models atau LLM) berbasis autoregresif pada seluruh domain komputasi keputusan telah memicu patologi arsitektural yang parah dalam sistem siber-fisik (cyber-physical systems/CPS) kritis-waktu dan perdagangan algoritmik sub-detik. Dekoder autoregresif memiliki keterlambatan inferensi intrinsik: pembentukan token demi token secara berurutan memaksakan latensi eksekusi temporal berorde $\mathcal{O}(N)$ dan saturasi lebar pita memori (memory bandwidth saturation), yang menimbulkan *State-Decision Drift* (pergeseran status-keputusan) katastropik di mana kondisi fisik atau pasar bergerak lebih cepat daripada siklus resolusi kebijakan kendali. Artikel ini menyajikan **DecisionModelBench**, sebuah kerangka pengujian empiris multi-domain yang mengevaluasi model-model keputusan non-autoregresif melawan LLM fondasional generatif (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, dan Gemma 2 2B) pada klaster komputasi terdistribusi dual NVIDIA Tesla T4 GPU dan infrastruktur cloud API. Kami merumuskan secara analitis integral Pergeseran Status Kontinu (*Continuous State Drift*), formula Pergeseran Harga Akibat Latensi (*Latency Slippage*: $P_{\text{fill}} = P_{\text{signal}}(1 \pm \gamma \sqrt{\Delta t / \tau_0})$), serta Pembusukan Alfa Eksponensial (*Negative Alpha Decay*: $\alpha(\Delta t) = \alpha_0 e^{-\lambda \Delta t}$). Melalui tiga lingkungan uji berlawanan—pertahanan udara taktis (klasifikasi multi-ancaman C-RAM/Iron Dome dengan kendala magazin baterai 20-rudal dan jeda *reload* 3,5 detik), eksekusi buku pesanan sub-detik (portofolio tiruan \$10.000), dan penangkisan proyektil dinamik kontinu (vektor kecepatan bola Brick Breaker)—model keputusan non-autoregresif (Laya 421M dengan latensi 55 ms, TypeSafe JEV 160 ms, OpenJev 0.5B 210 ms) secara konsisten mempertahankan integritas operasional dengan pemborosan nol token (zero token waste). Sebaliknya, LLM autoregresif (latensi 2.200–2.800 ms) berujung pada keruntuhan struktural: kehancuran kota total pada pertahanan udara, P&L negatif -\$19,41 berbanding +\$40,90 (Laya) pada perdagangan, dan kegagalan kendali 100% pada kinematika arkade. Kami secara ilmiah mendefinisikan batas mikrostruktur pasar—menegaskan bahwa perdagangan frekuensi sangat tinggi (UHFT) mikrodetik tetap menjadi ranah mutlak FPGA/C++, sementara model keputusan sub-detik mendominasi rentang 50–300 ms—dan membuktikan mengapa API cloud decision mengungguli LLM lokal 8B akibat hambatan bus memori sekuensial. Terakhir, kami merumuskan Arsitektur Kognitif Dua-Tingkat (*Decoupled Two-Tier Cognitive Architecture*) yang menyatukan penapisan refleks deterministik sub-100ms (Sistem 1) dengan penalaran strategis asinkron di luar jalur kritis (Sistem 2).

**Kata Kunci**—Sistem Siber-Fisik, Model Non-Autoregresif, Model Bahasa Besar Fondasional, Perdagangan Algoritmik, Pertahanan Udara, Kendali Waktu-Nyata, Pergeseran Latensi, Pembusukan Alfa, Arsitektur Dua-Tingkat.

---

## I. Pendahuluan

Perkembangan pesat kecerdasan buatan generatif telah melahirkan heuristika industri yang keliru: penerapan tanpa telaah kritis Model Bahasa Besar (Large Language Models atau LLM) autoregresif sebagai pengendali universal di berbagai subsistem perangkat lunak [1]. Dikenal luas sebagai paradigma *"LLM for everything"*, pola perancangan ini mengalirkan kueri persepsi terstruktur, penapisan (*triage*), perutean (*routing*), dan eksekusi waktu-nyata langsung menuju *pipeline* dekoder autoregresif dengan parameter antara 7 miliar hingga 70 miliar [2].

Meskipun model autoregresif menunjukkan pemahaman semantik yang mendalam pada dialog bahasa alami terbuka, sintesis dokumen luring, dan pembuatan kode program, pemanfaatannya sebagai pengendali lingkar-tertutup (*closed-loop controller*) pada sistem siber-fisik (CPS) kritis-waktu serta perdagangan algoritmik kuantitatif merepresentasikan kekeliruan arsitektural yang mendasar. Pembentukan teks autoregresif menghasilkan deret token secara sekuensial:
$$\Pr(y_{1:N} \mid x) = \prod_{i=1}^{N} \Pr(y_i \mid y_{<i}, x)$$
Akibatnya, pembentukan objek perutean berformat JSON atau instruksi eksekusi sepanjang $N \in [30, 250]$ token menuntut $N$ kali *forward pass* sekuensial yang dibatasi lebar pita memori (*memory-bandwidth bound*) melalui lapisan-lapisan dekoder Transformer. Sekalipun dikuantisasi ke presisi 4-bit (misalnya GGUF Q4\_K\_M) dan diakselerasi kernel CUDA, model 7B–9B pada akselerator standar industri (seperti NVIDIA Tesla T4) membutuhkan waktu komputasi antara 2.200 ms hingga lebih dari 6.000 ms per siklus keputusan.

Dalam teori kendali deterministik dan mikrostruktur finansial, waktu adalah koordinat fisik yang tak dapat diulang. Sistem fisik berevolusi secara kontinu mengikuti persamaan diferensial biasa:
$$\frac{d S(t)}{dt} = f(S(t), u(t), t)$$
Ketika sebuah agen mengamati status $S_t$ pada waktu $t$ namun memerlukan jeda inferensi $\Delta t_{\text{inference}} = t_{\text{act}} - t$ untuk menerbitkan aksi $u_t$, aksi tersebut tidak diterapkan pada status $S_t$, melainkan pada status $S_{t + \Delta t_{\text{inference}}}$. Jika laju divergensi lingkungan $\left\|\frac{\partial S}{\partial t}\right\|$ melampaui batas toleransi kendali, kebijakan yang dikeluarkan menjadi usang, tidak relevan, atau bahkan membawa kehancuran fisik—fenomena yang dalam riset ini kami definisikan sebagai **Pergeseran Status-Keputusan (*State-Decision Drift*)**.

Dalam perdagangan kuantitatif berkecepatan tinggi, keterlambatan eksekusi mengakibatkan pembusukan alfa negatif seketika dan seleksi merugikan (*adverse selection*). Pada pertahanan udara taktis, penundaan diskriminasi target mengakibatkan proyektil balistik menghantam target sipil sebelum rudal pencegat meluncur dari tabung peluncuran. Pada pelacakan gerak dinamis kontinu, kelambanan keputusan memicu kelaparan kendali (*control starvation*) dan kegagalan pencegatan total.

Untuk membuktikan dikotomi ini secara empiris, makalah ini menyajikan **DecisionModelBench**, platform tolok ukur terdistribusi yang membandingkan model keputusan non-autoregresif (Sistem 1) melawan LLM fondasional generatif (Sistem 2). Kontribusi utama penelitian ini meliputi:
1. **Perumusan Matematis Patologi Temporal:** Kami menyajikan penurunan analitis untuk Pergeseran Status Kontinu, Pergeseran Harga Akibat Latensi pada buku pesanan, dan Pembusukan Alfa Eksponensial, menetapkan batas teoretis ketidaklayakan inferensi autoregresif pada kontrol kritis.
2. **Lingkungan Uji Multi-Domain:** Kami mengimplementasikan tiga arena simulasi komprehensif: (i) Pertahanan Udara Taktis C-RAM/Iron Dome dengan kendala magazin pod 20 rudal, pendinginan *reload* 3,5 detik, dan perlindungan transponder IFF pesawat sipil; (ii) Perdagangan Algoritmik Sub-Detik dengan indikator teknikal streaming, ketidakseimbangan buku pesanan Level-2, dan model slippage kuadratik; serta (iii) Pelacakan Kinematika Dinamik Kontinu (intersepsi bola Brick Breaker).
3. **Evaluasi Terdistribusi Multi-GPU:** Kami menguji spektrum model keputusan—Laya 421M (ModernBERT RLCD), OpenJev 0.5B (skor logit kontinuasi), TypeSafe JEV System One (Cloud SaaS API), dan Kev 0.8B (LoRA)—berhadapan dengan LLM fondasional terbuka: Sahabat-AI 8B (LLM berdaulat Indonesia), Qwen 2.5 7B, dan Gemma 2 9B/2B, menggunakan *tensor-sharding* pada dual GPU NVIDIA Tesla T4.
4. **Verifikasi Empiris dan Batas Ilmiah:** Kami memaparkan data empiris keunggulan model non-autoregresif sub-100ms dengan nol token keluaran, sekaligus menguraikan batas ilmiah objektif antara ranah sub-detik model keputusan dan ranah mikrodetik perangkat keras FPGA/C++ ultra-HFT.
5. **Arsitektur Kognitif Dua-Tingkat Terkopel-Longgar (*Decoupled Two-Tier Cognitive Architecture*):** Kami memformulasikan pola desain asimetris yang memisahkan gerbang refleks deterministik sub-100ms dari penalaran mendalam LLM fondasional yang berjalan asinkron di luar jalur kritis.

---

## II. Kajian Terkait

### A. Model Bahasa Autoregresif dan Hambatan Dekoding Sekuensial
Model Bahasa Besar kontemporer bertumpu pada topologi Transformer *decoder-only* [3]. Saat inferensi dijalankan, proses komputasi terbagi menjadi dua fase: prapengisian (*prefill*) dan generasi autoregresif (*generation*). Prapengisian dapat diparalelkan di seluruh panjang token masukan $M$ (*compute-bound*), namun generasi token bersifat sekuensial dan terbatasi lebar pita memori (*memory-bandwidth bound*) [4]. Pada setiap pembentukan token $i \in [1, N]$, parameter bobot $\Theta$ beserta *Key-Value (KV) cache* harus ditransfer dari memori VRAM GPU ke inti komputasi (SRAM/ALU):
$$\text{Lalu Lintas Memori per Token} \approx 2 \cdot |\Theta| + 2 \cdot L \cdot d_{\text{model}} \cdot (M + i)$$
Untuk model 8 miliar parameter terkuantisasi 4-bit ($|\Theta| \approx 4,5 \text{ GB}$), pembuatan 32 token membutuhkan pemindahan lebih dari $144 \text{ GB}$ data melalui bus memori. Pada kartu akselerator seperti NVIDIA Tesla T4 yang memiliki lebar pita efektif teoritis $300 \text{ GB/s}$ (teramati $\sim 240 \text{ GB/s}$ di lapangan), laju generasi dibatasi secara fisik pada kisaran $\approx 30$ token/detik. Konsekuensinya, pembentukan respon JSON terstruktur membutuhkan waktu minimal $1.000 \text{ ms}$ hingga $2.500 \text{ ms}$.

Upaya optimasi seperti *speculative decoding* [5] dan arsitektur *medusa multi-head* [6] dapat menaikkan laju token rata-rata, namun tetap memperkenalkan ketidakpastian (*jitter*) latensi dan percabangan verifikasi spekulatif yang tidak dapat diterima dalam sistem kendali waktu-nyata keras (*hard real-time*).

### B. Enkoder Non-Autoregresif dan Pembobotan Logit Kontinuasi
Arsitektur saraf non-autoregresif menanggalkan siklus sekuensial demi pendekatan umpan-maju satu lintasan (*single-pass forward*) [7]. Representasi modern seperti ModernBERT [8] yang dipadukan dengan *Representation Learning via Contrastive Decoding* (RLCD) memungkinkan enkoder ringkas ($400\text{M} - 800\text{M}$ parameter) mengekstrak klasifikasi probabilitas, regresi kontinu, dan skor multikriteria secara simultan langsung dari tensor *embedding* terkontekstualisasi.

Pendekatan paralel lainnya memanfaatkan pembobotan logit kontinuasi satu-lintasan dari model kausal [9]. Alih-alih menghasilkan teks deskriptif, model memproses konteks masukan $X$ satu kali dan mengevaluasi distribusi logit tanpa normalisasi pada posisi token terakhir terhadap himpunan aksi kandidat $\mathcal{A} = \{a_1, a_2, \dots, a_k\}$:
$$P(a_k \mid X) = \frac{\exp(z_{a_k})}{\sum_{j=1}^{K} \exp(z_{a_j})}$$
Formulasi ini memangkas overhead komputasi dari $\mathcal{O}(N)$ transfer memori sekuensial menjadi operasi $\mathcal{O}(1)$, menghasilkan latensi deterministik antara 50 ms hingga 200 ms dengan probabilitas matematis terkalibrasi tanpa mengeluarkan satu pun token teks keluaran (*zero token waste*).

### C. Teori Kendali Sistem Siber-Fisik dan Mikrostruktur Pasar
Dalam teori kendali siber-fisik, stabilitas sistem kendali lingkar tertutup (seperti PID atau regulator kuadratik linier) bergantung pada margin fase (*phase margin*) [10]. Penambahan waktu tunda $\tau_d$ pada fungsi alih lingkar terbuka $G(s)$ memperkenalkan pergeseran fase negatif sebesar $\phi(\omega) = -\omega \tau_d$, yang mereduksi redaman sistem dan memicu osilasi tak terkendali hingga ketidakstabilan total.

Dalam ranah mikrostruktur finansial kuantitatif, model lambda Kyle [11] dan kerangka pangsa informasi Hasbrouck [12] membuktikan bahwa eksekusi pesanan agresif mengalami pergeseran harga yang berbanding lurus dengan latensi kedatangan pesanan di bursa. Buku pesanan elektronik batas (*limit order book*/LOB) modern menunjukkan laju pembatalan pesanan yang sangat tinggi, di mana masa hidup kuotasi likuiditas sering kali berada di bawah 100 milidetik [13]. Membebankan keputusan eksekusi pada LLM autoregresif multi-detik secara fundamental melanggar kaidah fisika mikrostruktur pasar.

---

## III. Formulasi Matematis dan Teori Kendali

Untuk memformalisasikan kegagalan model autoregresif dalam domain dinamis, kami memodelkan interaksi antara agen cerdas dan lingkungan stokastik kontinu.

```
+-------------------------------------------------------------------------+
|                  Status Lingkungan Kontinu S(t)                         |
+-------------------------------------------------------------------------+
       |                                                    ^
       | Pengamatan S_t                                     | Aksi Tertunda
       v                                                    | u(t + Delta_t)
+-----------------------------------+                       |
|   Jendela Pergeseran Status       |                       |
|   [ t  ---------->  t + Delta_t ] |                       |
+-----------------------------------+                       |
       |                                                    |
       |  Latensi Inferensi Delta_t                         |
       v                                                    |
+-----------------------------------------------------------+-------------+
| Sistem 1: Non-Autoregresif (Satu Forward Pass, 55 ms)     --> TEPAT SASARAN
| Sistem 2: LLM Autoregresif (Loop Token-per-Token, 2500ms) --> DRIFT / GAGAL
+-------------------------------------------------------------------------+
```

### A. Formulasi Pergeseran Status Kontinu (*Continuous State Drift*)
Misalkan vektor status lingkungan adalah $S(t) \in \mathbb{R}^d$, yang berevolusi berdasarkan persamaan diferensial stokastik:
$$dS(t) = f(S(t), u(t)) \, dt + \mathbf{\Sigma}(S(t)) \, dW(t)$$
di mana $f(\cdot)$ adalah dinamika *drift* deterministik, $u(t) \in \mathcal{U}$ adalah vektor kendali aksi, $\mathbf{\Sigma}(\cdot)$ adalah matriks difusi volatilitas, dan $W(t)$ adalah proses Wiener standar.

Agen memulai persepsi status pada penanda waktu $t_0$, mencuplik status $S_0 = S(t_0)$. Kebijakan $\pi_\theta$ memproses status tersebut untuk menghasilkan aksi $u = \pi_\theta(S_0)$. Evaluasi kebijakan memerlukan latensi komputasi $\Delta t = \Delta t_{\text{inference}} + \Delta t_{\text{network}}$. Aksi baru diaplikasikan pada penanda waktu $t_{\text{act}} = t_0 + \Delta t$.

Status riil lingkungan saat aksi tiba pada $t_{\text{act}}$ adalah:
$$S(t_{\text{act}}) = S_0 + \int_{t_0}^{t_0 + \Delta t} f(S(\tau), u_0) \, d\tau + \int_{t_0}^{t_0 + \Delta t} \mathbf{\Sigma}(S(\tau)) \, dW(\tau)$$
Kami mendefinisikan **Vektor Pergeseran Status-Keputusan (*State-Decision Drift Vector*)** $\mathbf{\delta}_S(\Delta t)$ sebagai deviasi Euclidean antara status yang diasumsikan agen ($S_0$) dan status aktual saat eksekusi ($S(t_{\text{act}})$):
$$\mathbf{\delta}_S(\Delta t) = S(t_{\text{act}}) - S_0 = \int_{t_0}^{t_0 + \Delta t} f(S(\tau), u_0) \, d\tau + \int_{t_0}^{t_0 + \Delta t} \mathbf{\Sigma}(S(\tau)) \, dW(\tau)$$

Nilai ekspektasi kuadrat pergeseran dievaluasi sebagai:
$$\mathbb{E}\left[\|\mathbf{\delta}_S(\Delta t)\|^2\right] = \left\|\int_{t_0}^{t_0 + \Delta t} f(S(\tau), u_0) \, d\tau\right\|^2 + \int_{t_0}^{t_0 + \Delta t} \text{Tr}\left(\mathbf{\Sigma}(S(\tau)) \mathbf{\Sigma}(S(\tau))^T\right) d\tau$$
Berdasarkan aproksimasi lokal drift konstan $\|f(S, u)\| \approx v_{\text{drift}}$ dan difusi isotropik $\mathbf{\Sigma} = \sigma \mathbf{I}$, relasi ini disederhanakan menjadi:
$$\mathbb{E}\left[\|\mathbf{\delta}_S(\Delta t)\|^2\right] \approx v_{\text{drift}}^2 (\Delta t)^2 + d \sigma^2 \Delta t$$

Ketika efikasi kendali mensyaratkan $\|\mathbf{\delta}_S\| < \epsilon_{\text{threshold}}$, terdapat batas cakrawala latensi kritis (*critical latency horizon*):
$$\Delta t_{\text{crit}} = \sup \left\{ \Delta t \;\middle|\; \mathbb{E}\left[\|\mathbf{\delta}_S(\Delta t)\|^2\right] \le \epsilon_{\text{threshold}}^2 \right\}$$
Jika $\Delta t_{\text{inference}} > \Delta t_{\text{crit}}$, kendali berbasis kebijakan lingkar-terbuka dipastikan mengalami instabilitas mendasar.

### B. Pergeseran Harga Akibat Latensi (*Latency-Induced Slippage*)
Dalam perdagangan buku pesanan batas (*limit order book*), sinyal beli/jual yang dibentuk pada harga tengah (*mid-price*) $P_{\text{signal}} = P(t_0)$ mengalami pergeseran harga selama transit jaringan dan komputasi inferensi. Kami memodelkan pergerakan harga tengah mikrostruktur sebagai gerak Brownian aritmetik dengan volatilitas $\sigma_{\text{market}}$:
$$P(t) = P(t_0) + \sigma_{\text{market}} \int_{t_0}^{t} dW(\tau)$$

Ketika pesanan pasar dieksekusi pada $t_{\text{act}} = t_0 + \Delta t$, pesanan tersebut menanggung biaya penyeberangan *spread* dan penalti seleksi merugikan (*adverse selection slippage*). Harga eksekusi efektif $P_{\text{fill}}$ untuk pesanan beli agresif dirumuskan sebagai:
$$P_{\text{fill}} = P_{\text{signal}} \left( 1 + \frac{\text{Spread}}{2 P_{\text{signal}}} + \gamma \sqrt{\frac{\Delta t}{\tau_0}} + \eta \left(\frac{Q}{V_{\text{book}}}\right)^\alpha \right)$$
di mana:
- $\gamma$ adalah koefisien mikrostruktur seleksi merugikan nir-dimensi;
- $\tau_0$ adalah konstanta normalisasi latensi acuan ($\tau_0 = 50 \text{ ms}$);
- $\eta \left(\frac{Q}{V_{\text{book}}}\right)^\alpha$ merepresentasikan dampak harga sesaat (Kyle-type impact) untuk ukuran pesanan $Q$ terhadap kedalaman kuotasi $V_{\text{book}}$.

Untuk pesanan jual agresif, formula slippage berlaku secara simetris ke arah bawah:
$$P_{\text{fill}} = P_{\text{signal}} \left( 1 - \frac{\text{Spread}}{2 P_{\text{signal}}} - \gamma \sqrt{\frac{\Delta t}{\tau_0}} - \eta \left(\frac{Q}{V_{\text{book}}}\right)^\alpha \right)$$

Saat latensi komputasi $\Delta t$ membengkak dari $\Delta t_{\text{fast}} = 55 \text{ ms}$ menjadi $\Delta t_{\text{LLM}} = 2.500 \text{ ms}$, rasio latensi $\frac{\Delta t}{\tau_0}$ melonjak 50 kali lipat, yang mengakibatkan kenaikan penalti slippage seleksi merugikan lebih dari $707\%$ ($\sqrt{50} \approx 7,07$).

### C. Dinamika Pembusukan Alfa Eksponensial (*Negative Alpha Decay*)
Alfa perdagangan $\alpha(t)$ merepresentasikan ekspektasi kelebihan imbal hasil (*excess return*) di atas tolak ukur pasar. Dalam pasar kompetitif, pelaku arbitrase menyerap inefisiensi harga dengan cepat, sehingga keunggulan informasi meluruh secara eksponensial terhadap waktu tunda eksekusi:
$$\alpha(\Delta t) = \alpha_0 e^{-\lambda \Delta t}$$
di mana $\alpha_0$ adalah keunggulan teoretis murni pada saat pengamatan awal ($t_0$) dan $\lambda > 0$ adalah konstanta laju pembusukan likuiditas buku pesanan.

Imbal hasil bersih terealisasi per transaksi, setelah dikurangi komisi bursa $C_{\text{fee}}$ dan slippage latensi $S(\Delta t) = \gamma \sqrt{\frac{\Delta t}{\tau_0}}$, dirumuskan sebagai:
$$R_{\text{net}}(\Delta t) = \alpha_0 e^{-\lambda \Delta t} - S(\Delta t) - C_{\text{fee}}$$

```
  Alfa / Imbal Hasil
    ^
    |  \alpha_0
    |   *
    |    \
    |     \   Cakrawala Imbal Hasil Bersih Positif
    |------\----------------------------------- Batas Impas (Break-Even)
    |       \
    |        \      P&L Negatif: R_net(Delta_t) < 0 (Seleksi Merugikan)
    |         *------------------------------->
   0+----------+-----------------------------+----> Latensi Delta_t
              50ms                         2500ms
           (Sistem 1)                    (Sistem 2 LLM)
```

Pada pasar aset kripto dan ekuitas yang sangat likuid, konstanta peluruhan empiris bernilai $\lambda \in [3,0, \, 15,0] \text{ s}^{-1}$. Untuk nilai moderat $\lambda = 5,0 \text{ s}^{-1}$:
- Pada $\Delta t = 55 \text{ ms}$: $\alpha(0,055) = \alpha_0 e^{-0,275} \approx 0,760 \alpha_0$ (76% alfa bertahan).
- Pada $\Delta t = 2.500 \text{ ms}$: $\alpha(2,5) = \alpha_0 e^{-12,5} \approx 3,7 \times 10^{-6} \alpha_0 \approx 0$ (Kepunahan total alfa).

Dengan demikian, sekalipun sebuah LLM autoregresif memiliki kecerdasan semantik makroekonomi yang luar biasa, model tersebut dipastikan menghasilkan kerugian finansial (P&L negatif) jika dipasang langsung pada pipa eksekusi waktu-nyata akibat keterlambatan yang menjebak ordermya pada zona seleksi merugikan.

---

## IV. Arsitektur DecisionModelBench & Domain Pengujian

Untuk membuktikan hipotesis ini secara empiris, kami merancang **DecisionModelBench**, platform tolok ukur komprehensif yang mengintegrasikan tiga domain simulasi dinamis.

### A. Domain 1: Pertahanan Udara Taktis (Simulator C-RAM / Iron Dome)
Pertahanan udara taktis merupakan sistem siber-fisik berisiko tinggi yang menuntut diskriminasi ancaman instan, kalkulasi intersepsi proyektil balistik, dan alokasi amunisi terbatas [17].

```
                           [ Radar Pertahanan Udara ]
                              (Ketinggian: 45 km)
                                       |
     Proyektil Musuh Masuk             |            Transponder IFF Sipil
  +-----------------------+            |       +-----------------------------+
  | Hipersonik (-35% HP)  |            |       | Garuda GA-402 (Squawk 7700) |
  | Meteorit   (-45% HP)  |            |       | Lion Air JT-610             |
  | Drone      (-12% HP)  |            |       +-----------------------------+
  +-----------------------+            |                      |
              \                        |                     /
               \                       |                    /
                v                      v                   v
      +---------------------------------------------------------------+
      |             Mesin Analisis & Keputusan Taktis AI              |
      +---------------------------------------------------------------+
                                       |
                 +---------------------+---------------------+
                 |                                           |
                 v                                           v
    [ Baterai Pod 20 Rudal ]                       [ SITREP Taktis Berkala ]
    - Jeda salvo ripple 2 ticks (100ms)            - Integritas Kota (HP)
    - Cooldown reload 3,5s (35 ticks)              - Serpihan Rendah (-3% HP)
```

1. **Tipologi Ancaman & Kalkulasi Kerusakan Fisik:**
   - **Rudal Balistik Hipersonik ($\mathcal{M} = 4,2 - 6,5$):** Proyektil bermanuver dengan pantulan radar kecil. Hantaman darat menyebabkan **-35,0% HP Kota**.
   - **Meteorit Impak Kinetik ($\mathcal{M} = 4,5 - 7,0$):** Proyektil massa tinggi berkecepatan terminal hiper. Hantaman darat menyebabkan **-45,0% HP Kota**.
   - **Drone Kamikaze ($\mathcal{M} = 0,3 - 0,8$):** Wahana nirawak lambat berketinggian rendah dengan hulu ledak terarah. Hantaman darat menyebabkan **-12,0% HP Kota**.
   - **Serpihan Ledakan Rendah (*Collateral Shrapnel*):** Jika rudal musuh berhasil diledakkan namun terjadi pada ketinggian di bawah $3,0 \text{ km}$ ($y > 280$ pada kanvas), serpihan ledakan menghujani kota dan menimbulkan **-3,0% HP Kota**.
   - **Pesawat Komersial Sipil:** Pesawat penumpang dengan transponder aktif (callsign `GARUDA-GA402`, `LION-JT610`, `CITILINK-QG801`, squawking IFF normal). Menembak pesawat sipil merupakan pelanggaran fatal (*friendly fire*) yang meruntuhkan skor integritas.
   - **Objek Nir-Bahaya:** Meteorit melintas orbit tinggi (`METEOR_MISS`) dan kawanan burung (`BIO-RCS-LOW`), yang wajib diabaikan demi menghemat amunisi baterai.

2. **Fisika Beban Baterai Realistis:**
   - **Kapasitas Magazin Pod:** Setiap pos pertahanan dibatasi 20 rudal pencegat per pod (mengacu pada standar tabung Tamir Iron Dome / C-RAM).
   - **Jeda Salvo Ripple:** Penembakan dibatasi jeda pelepasan minimal 2 ticks ($100 \text{ ms}$) guna mencegah pengosongan magazin instan.
   - **Siklus Cooldown Reload:** Saat amunisi habis (`0/20`), baterai memasuki status *RELOADING POD* selama **3,5 detik (35 ticks)**. Pada fase ini, kota sepenuhnya tidak berdaya terhadap serangan saturasi beruntun (*swarm saturation*).

3. **Panduan Navigasi Proporsional:**
   Rudal pencegat meluncur mengikuti kaidah navigasi proporsional:
   $$a_{\text{cmd}} = N' V_c \dot{\lambda}$$
   di mana $N'=3,5$ adalah faktor amplifikasi navigasi, $V_c$ kecepatan penutupan, dan $\dot{\lambda}$ laju sudut pandang garis bidik. Keterlambatan komputasi memperlambat peluncuran rudal, memaksa interceptor bermanuver tajam di akhir lintasan hingga kehilangan energi kinetik dan meleset dari sasaran.

### B. Domain 2: Perdagangan Algoritmik Sub-Detik
Arena simulasi finansial memodelkan buku pesanan aset kripto likuid (BTC/USDT spot) dengan modal awal portofolio **\$10.000,00 USD** untuk setiap model.
1. **Mikrostruktur & Sintesis Indikator Streaming:**
   - **Pita Ticker Sintetis Real-Time:** Menghasilkan pergerakan harga berbasis gerak Brownian geometrik yang dimodulasi regime: `NORMAL`, `BULL_RUN`, `FLASH_CRASH`, `SIDEWAYS`, dan `WHIPSAW`.
   - **Fitur Teknikal Streaming:** Kalkulasi bergerak untuk $\text{EMA}_9(t)$, $\text{EMA}_{21}(t)$, $\text{RSI}_{14}(t)$, dan Ketidakseimbangan Kedalaman Buku Pesanan Level-2:
     $$I_{\text{book}} = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}} \times 100\%$$
   - **Protokol Eksekusi:** Model mengevaluasi representasi status dan menerbitkan aksi diskret: `BUY`, `SELL`, atau `HOLD`.

2. **Kopling Latensi dan Slippage:**
   Harga pengisian pesanan (*fill price*) mengaplikasikan fungsi pergeseran kuadratik yang dirumuskan pada Bagian III-B:
   $$\text{Slippage}(\Delta t) = \min\left(0,035, \, 0,00025 \times \sqrt{\max\left(1,0, \, \frac{\Delta t}{50,0}\right)}\right)$$
   Untuk pesanan beli, $P_{\text{fill}} = P_{\text{signal}} \times (1 + \text{Slippage})$; untuk pesanan jual, $P_{\text{fill}} = P_{\text{signal}} \times (1 - \text{Slippage})$.

### C. Domain 3: Kinematika Arkade Dinamik Kontinu (Brick Breaker)
Untuk menguji stabilitas kendali lintasan mekanis kontinu, kami menerapkan arena simulasi arkade fisik.
1. **Mekanika Kinematika:**
   Bola bergerak dalam ruang koordinat 2D ($W = 14, H = 12$) dengan vektor kecepatan $\vec{v} = (v_x, v_y)$. Agen mengendalikan posisi horizontal pemukul (*paddle*) selebar $w_p = 4$.
2. **Cakrawala Batas Waktu Fisika:**
   Bola meluncur menuju garis dasar paddle pada $y_{\text{paddle}} = 11$. Jendela waktu penangkisan dibatasi secara ketat oleh hukum gerak:
   $$\Delta t_{\text{impact}} = \frac{y_{\text{paddle}} - y_{\text{ball}}}{v_y} \in [0,8, \, 1,8] \text{ detik}$$
   Untuk menangkis bola, pengendali harus memproyeksikan titik jatuh $x_{\text{proj}}$ dan menggeser paddle sebelum $\Delta t_{\text{impact}}$ terlampaui. Jika latensi inferensi melebihi rentang ini, paddle mengalami kelaparan perintah (*starvation*) dan bola menembus batas bawah arena.

---

## V. Pengaturan Eksperimental & Topologi Perangkat Keras

Seluruh pengujian empiris dijalankan pada sebuah klaster komputasi ganda enterprise yang terakselerasi GPU NVIDIA.

### A. Spesifikasi Node Komputasi
- **Prosesor Host:** Intel Xeon Processor (8 vCPUs @ 2,20 GHz), 32 GB RAM Sistem.
- **Akselerator Grafis:** Dual NVIDIA Tesla T4 GPU (masing-masing 15,6 GB GDDR6 VRAM, Arsitektur Turing, Compute Capability 7.5, TU104).
- **Driver & Stack CUDA:** NVIDIA Driver 580.82, CUDA Toolkit 12.8, PyTorch 2.5.1+cu124, runtime llama-cpp-python CUDA dengan kernel Flash Attention 2.0.

### B. Distribusi Model pada Multi-GPU Sharding
Untuk mencegah perebutan sumber daya komputasi (*resource contention*), pemetaan model diatur secara terdistribusi via `gpu_manager.py`:
- **GPU 0 (`cuda:0`, 15,6 GB VRAM):**
  - **Laya Multilingual (421M Parameter):** Enkoder dua arah ModernBERT dengan penyelarasan RLCD. Alokasi VRAM: $\sim 950 \text{ MB}$.
  - **Kev-0.8B (800M Parameter):** Arsitektur Qwen 2.5 dengan *pointer head* LoRA. Alokasi VRAM: $\sim 1.600 \text{ MB}$.
  - **Shard 0 LLM Fondasional:** Mengampu 50% pecahan tensor model generatif via parameter `tensor_split=[0.5, 0.5]`.
- **GPU 1 (`cuda:1`, 15,6 GB VRAM):**
  - **OpenJev (0.5B Parameter):** Model pembobot logit kontinuasi Qwen 2.5 satu-lintasan. Alokasi VRAM: $\sim 1.100 \text{ MB}$.
  - **Shard 1 LLM Fondasional:** Mengampu 50% sisa lapisan tensor model generatif.
- **Infrastruktur Cloud Terkelola (SaaS API):**
  - **TypeSafe JEV System One:** Endpoint API keputusan non-autoregresif komersial (`https://api.typesafe.ai/v1/systemone`) melalui koneksi aman TLS 1.3 / HTTP/2, tanpa membebani VRAM GPU lokal (0 MB).

### C. Spektrum LLM Fondasional Generatif (Sistem 2)
Seluruh LLM fondasional yang diuji menggunakan kuantisasi presisi 4-bit standar industri (GGUF Q4\_K\_M):
1. **Sahabat-AI 8B Instruct (8,03B parameter):** Model fondasional berdaulat Indonesia yang dikembangkan bersama oleh GoTo dan Indosat Ooredoo Hutchison [14]. Kapasitas memori: 4,6 GB.
2. **Qwen 2.5 7B Instruct (7,61B parameter):** Model instruksi multibahasa dari Alibaba Cloud [15]. Kapasitas memori: 4,4 GB.
3. **Gemma 2 9B Instruct (9,24B parameter):** Model penalaran terbuka dari Google DeepMind [16]. Kapasitas memori: 5,4 GB.
4. **Gemma 2 2B Instruct (2,61B parameter):** Model kelas ringkas berkecepatan tinggi dari Google. Kapasitas memori: 1,6 GB.

Model dijalankan pada temperatur rendah $T=0,1$ dengan format luaran JSON terstruktur:
`{"action": "BUY"|"SELL"|"HOLD", "urgency": "score", "confidence": float}`.

---

## VI. Hasil Empiris & Analisis Komparatif

### A. Kinerja Perdagangan Algoritmik Sub-Detik
Simulasi perdagangan algoritmik dijalankan sepanjang 100 ticks pasar simultan yang mencakup rezim lonjakan tren (`BULL_RUN`), keruntuhan mendadak (`FLASH_CRASH`), dan kondisi mendatar (`SIDEWAYS`). Seluruh model memulai pengujian dengan ekuitas awal **\$10.000,00 USD**. TABEL I menyajikan rangkuman hasil evaluasi finansial.

#### TABEL I: Kinerja Perdagangan Algoritmik Lintas Model Arsitektur (Portofolio \$10.000)
| Peringkat | Nama Model | Paradigma Arsitektur | Latensi ($\Delta t$) | Ekuitas Akhir | Total P&L | ROI (%) | Win Rate | Transaksi | Slippage/Order | Token Terbuang |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **Laya Multilingual (421M)** | Sistem 1 (ModernBERT RLCD) | **55 ms** | **\$10.040,90** | **+\$40,90** | **+0,41%** | **78,6%** | 14 | 0,025% | **0 token** |
| 🥈 **#2** | **TypeSafe JEV (Cloud API)** | Sistem 1 (Managed SaaS API) | **160 ms** | **\$10.033,46** | **+\$33,46** | **+0,33%** | **76,9%** | 13 | 0,044% | **0 token** |
| 🥉 **#3** | **OpenJev (0.5B Logits)** | Sistem 1 (Single Forward Pass) | **210 ms** | **\$10.030,85** | **+\$30,85** | **+0,31%** | **75,0%** | 12 | 0,051% | **0 token** |
| **#4** | **Kev-0.8B (Local LoRA)** | Sistem 1 (LoRA Pointer Head) | **950 ms** | **\$10.007,69** | **+\$7,69** | **+0,08%** | **55,6%** | 9 | 0,109% | **0 token** |
| **#5** | **Sahabat-AI 8B Instruct** | Sistem 2 (Autoregressive LLM) | **2.500 ms** | **\$9.980,59** | **-\$19,41** | **-0,19%** | **30,0%** | 10 | 0,177% | **320 token** |
| *Ref* | *Qwen 2.5 7B Instruct* | Sistem 2 (Autoregressive LLM) | 2.200 ms | \$9.983,88 | -\$16,12 | -0,16% | 33,3% | 9 | 0,166% | 288 token |
| *Ref* | *Gemma 2 9B Instruct* | Sistem 2 (Autoregressive LLM) | 2.800 ms | \$9.975,50 | -\$24,50 | -0,25% | 25,0% | 8 | 0,187% | 288 token |

```
Perbandingan Total P&L ($) Portofolio
+$50 +-----------------------------------------------------------------+
     |                                                                 |
+$40 |  [Laya: +$40,90]                                                |
     |        |                                                        |
+$30 |        +-- [JEV Cloud: +$33,46]                                 |
     |                  |                                              |
+$20 |                  +-- [OpenJev: +$30,85]                         |
     |                                                                 |
+$10 |                                  [Kev: +$7,69]                  |
     |                                                                 |
  $0 +-----------------------------------------------------------------+
     |                                                                 |
-$10 |                                                                 |
     |                                          [Qwen: -$16,12]        |
-$20 |                                                |   [Sahabat-AI: |
     |                                                |     -$19,41]   |
-$30 +------------------------------------------------+--------+-------+
    50ms              160ms     210ms          950ms 2200ms  2500ms
                                Latensi
```

Fakta empiris ini mengonfirmasi temuan analitis kami:
1. **Efisiensi Nol-Token Mutlak:** Model keputusan non-autoregresif menyelesaikan seluruh transaksi dengan **0 token luaran**, hanya mengeksekusi operasi kontraksi tensor satu-lintasan.
2. **Akumulasi Penalti Slippage:** Seluruh varian LLM autoregresif (Sahabat-AI, Qwen, Gemma) membukukan imbal hasil negatif (P&L minus). Keterlambatan 2,2–2,8 detik membuat order beli tereksekusi di puncak lonjakan harga (*buying at the top*), dan order jual tereksekusi di dasar jurang kepanikan (*selling at the bottom*).
3. **Daya Saing Endpoint Cloud:** TypeSafe JEV System One, kendati menempuh latensi bolak-balik jaringan internet ($\sim 100 \text{ ms}$ RTT), mampu menjaga siklus total di bawah 200 ms dan membukukan **+\$33,46 P&L**, jauh melampaui seluruh LLM lokal 8B.

### B. Hasil Simulasi Pertahanan Udara Taktis
Pengujian pertahanan udara mengevaluasi lima baterai pertahanan yang melindungi lima kota besar di Indonesia menghadapi tiga gelombang serangan proyektil (total 75 proyektil musuh dan 24 penerbangan komersial). TABEL II merangkum telemetri radar dan status akhir sektor kota.

#### TABEL II: Telemetri Radar Pertahanan Udara & Integritas Operasional (SITREP Gelombang 3)
| Kota & Sektor Baterai | Model Arsitektur AI | Latensi Kendali | Tingkat Pencegatan | Hantaman Lolos | Salah Tembak (Sipil) | Siklus Reload | Integritas Kota (HP) | Status Operasional | Konsumsi Token |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jakarta** | **Laya Multilingual (421M)** | **55 ms** | **96,0%** (24/25) | 1 (serpihan) | **0 / 8 (0,0%)** | 1 (aman) | **94,0%** | **OPERASIONAL PRIMA** | **0** |
| **Bandung** | **TypeSafe JEV (Cloud API)** | **160 ms** | **88,0%** (22/25) | 2 (1 drone, 1 serpihan) | **0 / 8 (0,0%)** | 1 (aman) | **88,0%** | **OPERASIONAL PRIMA** | **0** |
| **Surabaya** | **OpenJev (0.5B Logits)** | **210 ms** | **84,0%** (21/25) | 3 (1 rudal, 2 serpihan) | **0 / 8 (0,0%)** | 1 (aman) | **82,0%** | **OPERASIONAL PRIMA** | **0** |
| **Medan** | **Kev-0.8B (Local Ensemble)** | **950 ms** | **52,0%** (13/25) | 7 (2 rudal, 1 drone) | 1 / 8 (12,5%) | 1 (terekspos) | **42,0%** | **RUSAK WASPADA** | **0** |
| **Nusantara** | **Sahabat-AI 8B Instruct** | **2.500 ms** | **12,0%** (3/25) | 8 (2 meteor, 3 rudal) | 2 / 8 (25,0%) | 0 (runtuh) | **0,0%** | **KOTA HANCUR TOTAL** | **180** |
| *IKN (Alt 1)* | *Qwen 2.5 7B Instruct* | 2.200 ms | 16,0% (4/25) | 7 (2 meteor, 3 rudal) | 1 / 8 (12,5%) | 0 (runtuh) | 0,0% | KOTA HANCUR TOTAL | 180 |
| *IKN (Alt 2)* | *Gemma 2 9B Instruct* | 2.800 ms | 8,0% (2/25) | 8 (3 meteor, 3 rudal) | 2 / 8 (25,0%) | 0 (runtuh) | 0,0% | KOTA HANCUR TOTAL | 180 |

Analisis telemetri pertahanan udara menunjukkan patologi berikut:
1. **Kehancuran Fisik Pengendali Autoregresif:** Sektor Ibu Kota Nusantara (IKN) yang dilindungi Sahabat-AI 8B mengalami **kehancuran total (0,0% HP)** pada Tick 44 Gelombang 2. Rudal hipersonik melaju dengan laju $v_y \approx 4,2 \text{ px/tick}$. Selama proses pembuatan token kalimat penjelasan (~50 tick komputasi), rudal musuh telah menempuh lintasan lebih dari 210 piksel dan meratakan gedung kota sebelum perintah peluncuran rudal diterbitkan.
2. **Insiden Penembakan Pesawat Sipil:** Model autoregresif mengalami insiden salah tembak terhadap penerbangan Garuda GA-402 dan Lion JT-610. Hal ini disebabkan pergeseran koordinat radar selama siklus pembentukan token: ketika token aksi diterbitkan, vektor sasaran radar telah bergeser ke posisi pesawat sipil yang kebetulan melintas di dekat lintasan rudal awal.
3. **Manajemen Amunisi Baterai Pod:** Model keputusan cepat (Laya, JEV, OpenJev) mampu menembak proyektil musuh di ketinggian aman ($>15 \text{ km}$), sehingga magazin pod 20 rudal habis secara terkendali dan dapat diisi ulang (*reload* 3,5 detik) saat jeda antar gelombang. Sebaliknya, LLM lambat menimbun rudal karena terlambat berpikir, hingga baterai musnah dengan rudal masih tersisa di tabung peluncur.

### C. Kinematika Arkade Dinamik Kontinu (Brick Breaker)
Dalam simulasi arkade penangkisan bola kontinu sepanjang 140 ticks, kestabilan kendali mekanis diukur melalui skor balok dan kelangsungan hidup paddle. TABEL III merangkum hasil evaluasi tersebut.

#### TABEL III: Kinerja Kendali Kinematika Kontinu (Brick Breaker)
| Peringkat | Nama Model | Latensi Inferensi | Frekuensi Siklus | Skor Balok | Status Akhir | Sisa Nyawa | Token Terbuang |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **Laya Multilingual (421M)** | **55 ms** | **18,2 Hz** | **240 poin** | **BERTAHAN (MENANG)** | **3 / 3** | **0** |
| 🥈 **#2** | **TypeSafe JEV (Cloud API)** | **160 ms** | **6,2 Hz** | **210 poin** | **BERTAHAN (STABIL)** | **3 / 3** | **0** |
| 🥉 **#3** | **OpenJev (0.5B Logits)** | **210 ms** | **4,8 Hz** | **190 poin** | **BERTAHAN (STABIL)** | **2 / 3** | **0** |
| **#4** | **Kev-0.8B (Local LoRA)** | **950 ms** | **1,05 Hz** | **70 poin** | **RUSAK TERBENTUR** | **1 / 3** | **0** |
| **#5** | **Sahabat-AI 8B Instruct** | **2.500 ms** | **0,40 Hz** | **10 poin** | **GUGUR (LAG FISIKA)** | **0 / 3 (Tick 28)** | **144** |

Dengan batas waktu jatuh $\Delta t_{\text{impact}} \approx 1,2 \text{ detik}$, Laya 421M mengeksekusi lebih dari 21 kali pembaruan kendali arah, menjaga posisi paddle tepat di bawah bola. Sahabat-AI 8B hanya mampu memperbarui keputusan setiap 2,5 detik (0,4 Hz), sehingga bola telah jatuh bebas menembus lantai sebelum kata pertama selesai digenerasi (gugur pada Tick 28).

---

## VII. Diskusi, Batasan Ilmiah, & Realitas Ranah Mikrodetik

Untuk menjaga kejujuran ilmiah, kami mendefinisikan batasan operasional model keputusan non-autoregresif secara tegas dan meluruskan kesalahpahaman yang jamak terjadi dalam industri komputasi.

```
+-------------------------------------------------------------------------------+
|                    SPEKTRUM LATENSI OPERASIONAL KOMPUTASI                     |
+-------------------------------------------------------------------------------+
  Ranah Mikrodetik (UHFT)       Ranah Keputusan Sub-Detik       Ranah Makro Strategis
  [ 1 us - 500 us ]             [ 20 ms - 300 ms ]               [ 2 s - 60 s+ ]
+-------------------------+  +--------------------------+  +--------------------+
| Perangkat Keras Murni   |  | Model Keputusan          |  | LLM Fondasional    |
| - Custom ASIC / FPGA    |  | Non-Autoregresif         |  | Autoregresif       |
| - C++ Kernel Bypass     |  | - Laya (55ms), JEV(160ms)|  | (Sistem 2)         |
|   (Solarflare OpenOnload|  | - Satu Forward Pass      |  | - Analisis Sentimen|
| - Kolokasi Bursa        |  | - Imbalance Buku Pesanan |  | - Sintesis Laporan |
|   (Cross-Connect Direct)|  |   dan Triage Instan      |  | - Diplomasi Empatik|
+-------------------------+  +--------------------------+  +--------------------+
```

### A. Ranah Mikrodetik: Perdagangan Frekuensi Sangat Tinggi (UHFT)
Klaim bahwa model kecerdasan buatan berbasis *neural network* berlatensi 50–200 ms dapat digunakan untuk *Ultra-High-Frequency Trading* (UHFT) adalah keliru secara ilmiah [18]. Pada arena bursa derivatif kelas satu (misalnya CME, NASDAQ, Binance cross-connect di Equinix LD4/NY4):
1. **Perebutan Antrean Mikrodetik:** Persaingan antrean kuotasi likuiditas berlangsung dalam skala $1 \text{ hingga } 50 \text{ mikrodetik}$ ($\mu\text{s}$). Ranah ini sepenuhnya dikuasai sirkuit keras terdedikasi: FPGA (*Field-Programmable Gate Arrays*), ASIC (*Application-Specific Integrated Circuits*), serta *driver* jaringan *kernel-bypass* (Solarflare OpenOnload / EF\_VI) berbasis kode C++ murni.
2. **Batasan Fisika Bus Komputasi:** Waktu transfer data melalui bus PCIe kartu grafis atau *socket buffer* sistem operasi telah memakan waktu $\sim 5 - 15 \mu\text{s}$, melebihi keseluruhan jendela eksekusi UHFT. Tidak ada arsitektur *deep learning* yang dapat beroperasi pada domain mikrodetik ini.

### B. "Sweet Spot" Model Keputusan: Eksekusi Sub-Detik (50 ms – 300 ms)
Model keputusan non-autoregresif menempati takhta ideal pada lapisan **eksekusi algoritmik sub-detik** ($20 \text{ ms} - 500 \text{ ms}$). Domain ini meliputi:
- Arbitrase statistik kuantitatif dan perutean pesanan cerdas (*smart order routing*);
- Penapisan transaksi mencurigakan dan pencegahan penipuan finansial seketika;
- Sistem telemetri keamanan dan kendali interlock siber-fisik;
- Klasifikasi kueri komplain pelanggan instan dan eskalasi otomatis.

Pada rentang waktu ini, ruang status terlalu rumit untuk diselesaikan oleh sekadar aturan baku statis (*if-else*), namun jendela waktu respons melarang keras beban latensi dekoder LLM autoregresif.

### C. Mengapa Cloud Decision API Mengungguli LLM Lokal 8B pada GPU
Salah satu temuan menarik dari Tabel I dan II adalah bahwa **TypeSafe JEV System One** (API cloud yang diakses via jaringan internet dengan latensi bolak-balik $\sim 160 \text{ ms}$) secara konsisten mengalahkan model lokal 8B yang berjalan langsung pada GPU Tesla T4 fisik.

Hal ini dapat dijelaskan secara matematis melalui pemisahan kompleksitas komputasi:
- **LLM Autoregresif Lokal (8B):**
  $$\text{Latensi} = \sum_{i=1}^{N} \frac{2 \cdot |\Theta|_{\text{LLM}}}{\text{Lebar Pita Memori}} \approx N \times \frac{2 \times 4,5 \text{ GB}}{240 \text{ GB/s}} \approx N \times 37,5 \text{ ms}$$
  Untuk jawaban sepanjang $N = 60$ token, latensi lokal GPU melampaui $2.250 \text{ ms}$, terkunci oleh kecepatan transfer bus memori.
- **Model Keputusan Cloud (JEV API):**
  $$\text{Latensi} = t_{\text{DNS}} + t_{\text{TLS}} + t_{\text{RTT}} + t_{\text{single-pass forward}} \approx 1 \text{ ms} + 2 \text{ ms} + 150 \text{ ms} + 5 \text{ ms} = 158 \text{ ms}$$
Karena model cloud mengevaluasi logit keputusan dalam satu kali *forward pass* tanpa loop generasi token, total waktu responsnya didominasi oleh perambatan foton serat optik antar pulau, yang tetap jauh lebih cepat daripada bottleneck memori sekuensial pada GPU lokal saat menjalankan model generatif besar.

---

## VIII. Arsitektur Kognitif Dua-Tingkat Terkopel-Longgar & Kesimpulan

### A. Arsitektur Kognitif Dua-Tingkat (*Decoupled Two-Tier Cognitive Architecture*)
Untuk menyatukan kecepatan refleks kritis-waktu dengan kedalaman penalaran model bahasa besar, kami merumuskan **Arsitektur Kognitif Dua-Tingkat Terkopel-Longgar**:

```
                       [ Aliran Data Sensor / Kueri Masuk ]
                                       |
                                       v
                   +---------------------------------------+
                   |  TIER 1: MESIN REFLEKS (Sistem 1)     |
                   |  Model Keputusan Non-Autoregresif     |
                   |  - Laya / OpenJev / TypeSafe JEV      |
                   |  - 1x Forward Pass CUDA (0 Token)     |
                   |  - Latensi Deterministik: 50 - 160 ms |
                   +---------------------------------------+
                                       |
                       Evaluasi Ambang Batas Probabilitas
                                       |
              +------------------------+------------------------+
              |                                                 |
     [ Di Bawah Ambang Kritis ]                        [ Butuh Penalaran Dalam ]
     Jalur Cepat (Fast-Path)                           Pesan Asinkron (Event Bus)
              |                                                 |
              v                                                 v
+---------------------------+                   +-------------------------------+
| Eksekusi Langsung / Aksi  |                   | TIER 2: MESIN MUSYAWARAH      |
| - Luncurkan Rudal Tangkis |                   | LLM Fondasional Generatif     |
| - Eksekusi Order Saham/BTC|                   | - Sahabat-AI / Qwen / Gemma   |
| - 100% Kuota LLM Dihemat  |                   | - Analisis Narasi & Sentimen  |
+---------------------------+                   | - Latensi: 2.000 - 5.000 ms   |
                                                +-------------------------------+
                                                                |
                                                                v
                                                +-------------------------------+
                                                | Rekomendasi Strategi Makro    |
                                                | Evaluasi Pasca-Insiden        |
                                                | Pembaruan Ambang Batas Tier 1 |
                                                +-------------------------------+
```

1. **Tier 1: Mesin Refleks (Sistem 1):**
   - **Karakteristik:** Model enkoder padat atau penilai logit kontinuasi (Laya 421M, OpenJev 0.5B, atau TypeSafe JEV API).
   - **Cakrawala Operasional:** $\Delta t \le 100 \text{ ms}$, 0 token keluaran, probabilitas matematis terkalibrasi (tensor Sigmoid/Softmax).
   - **Tanggung Jawab:** Aktuasi mekanis instan, eksekusi kuotasi buku pesanan, intersepsi ancaman udara, dan penapisan gerbang keselamatan (*safety gating*).
2. **Tier 2: Mesin Musyawarah (Sistem 2):**
   - **Karakteristik:** Model fondasional generatif (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B).
   - **Cakrawala Operasional:** $\Delta t \in [2, 10] \text{ detik}$, terhubung melalui antrean pesan asinkron (*message queue* seperti Redis/Kafka).
   - **Tanggung Jawab:** Investigasi akar penyebab insiden pasca-kejadian, analisis tren makroekonomi, diplomasi komunikasi empatik, dan pembaruan parameter kebijakan strategis.

Dengan pola terpisah ini, 80% hingga 90% kejadian operasional diselesaikan instan pada Tier 1 di bawah 100 ms, membebaskan klaster LLM dari kejenuhan komputasi dan memitigasi risiko *state-decision drift*.

### B. Kesimpulan
Hasil riset empiris dalam DecisionModelBench membuktikan secara konklusif ketidaklayakan Model Bahasa Besar autoregresif sebagai pengendali langsung pada sistem kritis-waktu. Keterbatasan lebar pita memori pada proses generasi token berurutan memicu kelambanan respon multi-detik yang berujung pada pergeseran status-keputusan, kerugian slippage finansial yang fatal, dan kegagalan pertahanan fisik.

Model keputusan non-autoregresif menyelesaikan kebuntuan ini dengan menyusun ulang inferensi ke dalam satu lintasan maju, memberikan keluaran matematis deterministik di bawah 100 milidetik dengan nol pemborosan token. Meskipun eksekusi mikrodetik murni tetap menjadi kedaulatan perangkat keras FPGA/ASIC, model keputusan non-autoregresif merupakan fondasi arsitektur terbaik untuk sistem siber-fisik dan perdagangan algoritmik sub-detik. Infrastruktur otonom masa depan wajib mengadopsi topologi kognitif dua-tingkat guna menjamin kelangsungan operasional fisik dan keberlanjutan ekonomi.

---

## Ucapan Terima Kasih
Penulis menyampaikan apresiasi mendalam kepada komunitas kecerdasan buatan sumber terbuka, pengembang ModernBERT, tim Qwen, tim Gemma, tim pengembang Sahabat-AI, serta tim periset TypeSafe AI atas akses pengujian API selama tolak ukur ini dilaksanakan.

---

## Referensi

1. J. Achiam *et al.*, "GPT-4 technical report," *arXiv preprint arXiv:2303.08774*, 2023.
2. A. Touvron *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
3. A. Vaswani *et al.*, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.
4. R. Y. Aminabadi *et al.*, "DeepSpeed-inference: Enabling efficient inference of Transformer models at unprecedented scale," in *IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2022, pp. 1–15.
5. C. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *International Conference on Machine Learning (ICML)*, 2023, pp. 19274–19286.
6. T. Cai *et al.*, "Medusa: Simple LLM inference acceleration with multiple decoding heads," *arXiv preprint arXiv:2401.10774*, 2024.
7. J. Gu, J. Bradbury, C. Xiong, V. O. Li, and R. Socher, "Non-autoregressive neural machine translation," in *International Conference on Learning Representations (ICLR)*, 2018.
8. B. Warner *et al.*, "ModernBERT: Bringing BERT into the modern era of deep learning," *Answer.AI & LightOn Technical Report*, 2024.
9. TypeSafe AI Research, "System One: Sub-200ms non-autoregressive decision architectures for enterprise triage," *TypeSafe Whitepaper*, 2025.
10. K. J. Åström and R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton, NJ: Princeton University Press, 2021.
11. A. S. Kyle, "Continuous auctions and informed trader," *Econometrica*, vol. 53, no. 6, pp. 1315–1335, 1985.
12. J. Hasbrouck, *Empirical Market Microstructure: The Institutions, the Economics, and the Econometrics of Securities Trading*. Oxford, UK: Oxford University Press, 2007.
13. M. O’Hara, "High frequency market microstructure," *Journal of Financial Economics*, vol. 116, no. 2, pp. 257–270, 2015.
14. GoTo and Indosat Ooredoo Hutchison, "Sahabat-AI: Indonesian sovereign large language models," *Technical Whitepaper*, 2024.
15. Qwen Team, "Qwen2.5 technical report," *Alibaba Cloud Technical Report*, 2024.
16. Gemma Team, "Gemma 2: Improving open language models at a practical size," *Google DeepMind Technical Report*, 2024.
17. P. A. Zandbergen, "Accuracy of air defense systems under high-velocity multi-threat saturation," *IEEE Transactions on Aerospace and Electronic Systems*, vol. 58, no. 4, pp. 3120–3134, 2022.
18. E. Budish, P. Cramton, and J. Shim, "The high-frequency trading arms race: Frequent batch auctions as a market design response," *Quarterly Journal of Economics*, vol. 130, no. 4, pp. 1547–1621, 2015.
