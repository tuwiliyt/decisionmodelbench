# Penapisan Niat Deterministik, Panggilan Darurat Berdaulat 112/911, dan Triase Sipil Dua-Tingkat: Evaluasi Empiris Model Keputusan Non-Autoregresif Melawan Model Bahasa Besar Fondasional pada Operasi Penyelamatan Nyawa dan Administrasi Publik Bervolume Tinggi

**Penulis:** Richie O. Sumual  
*Advanced Agentic AI & Distributed Systems Research, Jakarta, Indonesia*  
*Kontak:* `richie@panita.web.id`  
*Format Naskah:* IEEE Transactions on Computational Social Systems / IEEE Transactions on Services Computing  
*Identitas Laporan:* Riset Mandiri — Richie O. Sumual (Riset Mandiri Ke-3, September 2026)

---

### Abstrak
Integrasi masif Model Bahasa Besar (*Large Language Models*/LLM) autoregresif ke dalam kanal tanggap darurat terpadu pemerintah daerah (Panggilan Darurat 112 Indonesia / 911), kanal disposisi administrasi publik nasional (SP4N-LAPOR!), serta pusat layanan pelanggan korporat telah menyingkap kerentanan operasional yang kritis: penumpukan antrean ekstrem, kegagalan pemenuhan kesepakatan tingkat layanan (*Service Level Agreement*/SLA), mutasi sintaksis skema data JSON yang non-deterministik, serta risiko kebocoran data pribadi warga (*Personally Identifiable Information*/PII). Pada triase darurat penyelamatan nyawa, sintesis teks sekuensial token-demi-token menimbulkan latensi multi-detik ($>3.300$~ms hingga $>12.600$~ms) yang memangkas secara fatal batas klinis "periode emas" (*golden period*) 3 hingga 5 menit pada kasus henti jantung di luar rumah sakit serta melanggar regulasi batas waktu pengiriman armada ($<3$ detik). Dalam makalah ini, kami menyajikan evaluasi empiris menggunakan tolok ukur **DecisionModelBench**, yang menguji model keputusan non-autoregresif melawan jajaran LLM generatif fondasional unggulan dunia—meliputi model kedaulatan Indonesia Sahabat-AI 8B Instruct, Qwen 2.5 7B Instruct dari Alibaba Cloud, dan Gemma 2 9B Instruct dari Google DeepMind pada peladen akselerator ganda NVIDIA Tesla T4—melintasi lima belas skenario komprehensif: sepuluh skenario sipil dan korporat serta lima skenario kritis panggilan darurat 112/911. Kami memformulasikan dinamika ledakan antrean Erlang-C, integral probabilitas pelanggaran batas waktu SLA, fungsi peluruhan probabilitas kelangsungan hidup klinis pasien henti jantung, serta hukum entropi kegagalan skema JSON. Secara empiris, model keputusan non-autoregresif (TypeSafe JEV System One pada 197,6 ms, OpenJev 0,5B pada 560,4 ms, Kev-0,8B pada 648,5 ms, dan Laya 421M terisolasi pada 35,8–63,2 ms) mencapai 100% kepatuhan skema, akurasi perutean semantik hingga 100%, serta nol emisi token dengan biaya \$0,05 per 100.000 transaksi. Sebaliknya, LLM Mandiri menunjukkan latensi pengiriman armada yang berbahaya: 3.340,0 ms untuk Sahabat-AI 8B, 10.024,4 ms untuk Qwen 2.5 7B, dan 12.681,7 ms untuk Gemma 2 9B, di mana Qwen mengalami kerapuhan sintaksis fatal (akurasi mandiri anjlok ke 33,3% akibat pembungkusan tanda markdown). Menjawab krisis ini, kami mengevaluasi **Pipeline Tandem Hibrida Dua-Tingkat** (*Two-Tier Hybrid Brain Pipeline*) di mana Tingkat 1 (Model Keputusan JEV) mengeksekusi disipasi sinyal pengiriman armada penyelamat secara instan dalam 178,5 ms hingga 217,7 ms ($<$220 ms, akselerasi hingga 64,2$\times$, akurasi 100%, 0 token), sementara Tingkat 2 secara paralel/asinkron mengasimilasi metadata triase guna menyintesis panduan verbal resusitasi jantung paru (RJP) yang mendalam (207–294 token). Konfigurasi tandem ini mencapai kondisi Pareto Optimal—memadukan kecepatan fisik armada instan dengan bimbingan pertolongan pertama yang menenangkan warga.

**Kata Kunci:** Kecerdasan Buatan Sektor Publik, Kedaulatan AI, Panggilan Darurat 112/911, Pipeline Tandem Hibrida Dua-Tingkat, Penapisan Niat, Model Keputusan Non-Autoregresif, Model Bahasa Besar, Antrean Erlang-C, Resusitasi Periode Emas, Pelindungan Data Pribadi, Total Biaya Kepemilikan.

---

## I. Pendahuluan
Pemerintah daerah, kementerian/lembaga penanggulangan bencana, kepolisian, dan pusat layanan pelanggan korporat kian mengadopsi kecerdasan buatan guna mengelola lonjakan volume laporan, pengaduan, dan permintaan bantuan dari masyarakat [1], [2]. Kanal layanan publik digital nasional di Indonesia—seperti layanan panggilan darurat terpadu 112, Sistem Pengelolaan Pengaduan Pelayanan Publik Nasional (SP4N-LAPOR!), serta sistem komando operasi penanggulangan bencana (BPBD dan Basarnas)—kerap menghadapi lonjakan lalu lintas yang drastis saat terjadi bencana hidrometeorologi, krisis keamanan sipil, maupun kecelakaan transportasi massal. Pada pusat kendali panggilan darurat, waktu merupakan variabel mutlak penentu keselamatan jiwa dan penyelamatan aset: pada henti jantung mendadak di luar rumah sakit (*out-of-hospital cardiac arrest*) atau henti napas akut, keterlambatan setiap tiga puluh detik dalam memberangkatkan ambulans paramedis ICU menurunkan peluang resusitasi secara eksponensial [14], [15].

Guna mengotomatisasi penerimaan, penapisan niat (*intent gating*), dan perutean disposisi pengaduan secara terstruktur, pengembang sistem kerap mengandalkan Model Bahasa Besar (*Large Language Models*/LLM) autoregresif [3], [4], [16]. Dalam pola perancangan ini, transkrip suara penelepon dimasukkan ke dalam templat instruksi (*prompt*), dan model generatif berparameter 7 hingga 70 miliar parameter diinstruksikan untuk menyintesis muatan terstruktur JSON token demi token:
$$\Pr(y_{1:L} \mid X) = \prod_{k=1}^{L} \Pr(y_k \mid y_{<k}, X)$$
Meskipun LLM autoregresif menunjukkan keluwesan linguistik yang mengesankan, penggunaannya sebagai penapis langsung pada gerbang depan operasional panggilan darurat memicu patologi arsitektural yang berbahaya:
1. **Latensi Pengiriman Armada Kritis dan Ledakan Antrean:** Pembentukan struktur JSON menuntut $L \in [150, 300]$ langkah inferensi sekuensial yang dibatasi oleh lebar pita memori (*memory bandwidth*) pada akselerator GPU. Pada peladen kelas komersial, hal ini menimbulkan latensi inferensi antara 3.340 ms hingga 12.680 ms per laporan. Regulasi darurat (seperti NFPA 1221, standar EENA, dan ketentuan Kementerian Komunikasi dan Informatika untuk layanan 112) menetapkan bahwa penentuan disposisi dan peluncuran armada awal wajib diselesaikan di bawah 3 detik ($\tau_{\text{dispatch}} < 3{,}0$ dtk). LLM autoregresif secara inheren melanggar ambang batas ini. Terlebih lagi, dalam dinamika antrean Erlang-C, durasi layanan multi-detik ini memicu ledakan antrean eksponensial saat krisis terjadi, menipiskan alokasi *socket* jaringan dan menelantarkan penelepon dalam bahaya maut.
2. **Entropi Skema dan Kerapuhan Sintaksis JSON:** Pengambilan sampel token secara autoregresif bersifat stokastik. Akumulasi probabilitas pergeseran token (*token drift*) pada cakrawala generasi yang panjang kerap memicu kelalaian tanda kurung, tanda koma, atau pembungkusan tanda markdown (\texttt{```json}), sehingga sistem otomatis *Computer-Aided Dispatch* (CAD) hilir gagal melakukan parsing data.
3. **Pemborosan Energi dan Biaya Operasional (TCO):** Pembangkitan 200 token per kueri pada ratusan ribu transaksi harian menuntut daya listrik komputasi GPU yang masif, membebani anggaran fiskal publik secara tidak rasional.
4. **Pelanggaran Kedaulatan Data dan Risiko PII:** Memasukkan data sensitif warga—seperti Nomor Induk Kependudukan (NIK), nomor rekening bank, rekam medis, dan alamat domisili—ke dalam jendela atensi autoregresif menimbulkan risiko retensi memori dan kebocoran data, yang melanggar ketentuan Undang-Undang Pelindungan Data Pribadi (UU PDP No. 27 Tahun 2022). Selain itu, model fondasional generatif rentan terhadap manipulasi perintah (*prompt injection*) yang dapat mengekstrak kredensial sistem internal.

Untuk mengatasi keterbatasan tersebut, makalah ini meneliti penerapan **model keputusan non-autoregresif** sebagai gerbang refleks semantik berlatensi sub-200ms (Sistem 1) dalam administrasi publik, penanganan panggilan darurat, dan layanan korporat. Berbeda dari dekoder generatif, model keputusan non-autoregresif memproyeksikan representasi teks aduan secara langsung ke ruang aksi kategori dan tensor urgensi dalam satu lintasan maju ($\mathcal{O}(1)$), menghasilkan struktur deterministik dengan konsumsi nol token generasi.

Lebih jauh, menyadari bahwa penelepon darurat membutuhkan peluncuran armada fisik yang seketika sekaligus panduan suara yang menenangkan, kami mengevaluasi **Pipeline Tandem Hibrida Dua-Tingkat** (*Two-Tier Hybrid Brain Pipeline*) yang diimplementasikan pada peladen multi-GPU terdistribusi (akselerator ganda NVIDIA Tesla T4). Pada rancangan tandem ini, Tingkat 1 (Model Keputusan) mengeksekusi triase deterministik sub-200ms untuk memberangkatkan armada seketika tanpa token generasi, sementara Tingkat 2 (Sahabat-AI 8B, Qwen 2.5 7B, atau Gemma 2 9B) secara paralel/asinkron mengasimilasi metadata triase terstruktur guna menyintesis protokol pertolongan pertama (seperti irama kompresi dada RJP atau rute evakuasi asap gedung).

Melalui tolok ukur **DecisionModelBench**, kami menyajikan evaluasi empiris komprehensif yang membandingkan model keputusan non-autoregresif (TypeSafe JEV System One, OpenJev 0,5B, Kev-0,8B, dan Laya Multilingual 421M) dengan model generatif fondasional (Sahabat-AI 8B Instruct, Qwen 2.5 7B Instruct, dan Gemma 2 9B Instruct). Kontribusi utama dari karya ilmiah ini adalah:
- **Formulasi Teoretis Antrean, SLA, dan Resusitasi Klinis:** Kami merumuskan dinamika waktu tunggu antrean Erlang-C dan membuktikan secara analitis mengapa pergeseran waktu layanan dari 197,6 ms ke 5.107,8 ms memicu ledakan antrean tak hingga saat krisis publik terjadi. Kami memformulasikan integral probabilitas pelanggaran SLA, persamaan entropi kegagalan skema sintaksis, serta fungsi peluruhan probabilitas kelangsungan hidup pasien henti jantung pada periode emas klinis.
- **Tolok Ukur Lima Belas Skenario Sipil dan Panggilan Darurat:** Kami mengevaluasi 10 skenario administrasi publik dan korporat serta 5 skenario kritis panggilan darurat 112/911.
- **Pengujian Tandem Lintas Arsitektur pada Perangkat Keras Berdaulat:** Kami memvalidasi kinerja Pipeline Tandem Hibrida pada akselerator ganda NVIDIA Tesla T4 melintasi tiga LLM unggulan (Sahabat-AI 8B, Qwen 2.5 7B, dan Gemma 2 9B). Tingkat 1 memberangkatkan armada penyelamat dalam 178,5 ms hingga 217,7 ms ($<$220 ms, akselerasi hingga 64,2$\times$, 0 token), sementara Tingkat 2 menyintesis bimbingan verbal mendalam dalam 207–294 token, membuktikan tercapainya kondisi Pareto Optimal.
- **Pelepasan Beban Komputasi Panggilan Hoaks/Iseng:** Kami membuktikan bahwa model keputusan menyaring panggilan prank dalam 223,5 ms tanpa membuang token generasi GPU, mencegah kelebihan beban komputasi antrean saat bencana massal terjadi.
- **Evaluasi Kinerja, Efisiensi Biaya, dan Tata Kelola Privasi:** Kami membuktikan bahwa model keputusan mencapai kepatuhan skema 100%, akurasi perutean hingga 100%, dan latensi sub-200ms dengan biaya \$0,05 per 100.000 transaksi. Kami mengungkap kerapuhan sintaksis LLM mandiri (seperti Qwen 2.5 yang anjlok ke akurasi 33,3% akibat galat parsing tanda markdown) dan membuktikan bahwa model keputusan bertindak sebagai benteng semantik deterministik yang mematuhi UU PDP No. 27/2022 sembari memangkas biaya komputasi sebesar 99,87%.

---

## II. Tinjauan Pustaka

### A. Model Bahasa Autoregresif dalam Administrasi Publik
Model Bahasa Besar berbasis arsitektur Transformer khusus-dekoder (*decoder-only*) [5] telah banyak dieksplorasi untuk pemrosesan dokumen otomatis, agen percakapan publik, dan klasifikasi teks [1], [2]. Guna menghadirkan kecerdasan buatan yang berdaulat secara budaya dan bahasa, inisiatif riset nasional di Indonesia telah melahirkan model Sahabat-AI [3], yang dilatih menggunakan korpora bahasa Indonesia dan bahasa daerah dalam skala miliaran token.

Kendati demikian, pengoperasian LLM fondasional pada jalur perutean menghadapi kendala fisik transfer data memori (*memory-bandwidth bound*) [6]. Setiap token yang dihasilkan mengharuskan seluruh parameter bobot $\Theta$ dimuat dari memori VRAM menuju unit komputasi. Untuk model berparameter 8 miliar pada kuantisasi 4-bit ($|\Theta| \approx 4{,}5$ GB), sintesis muatan JSON sepanjang 200 token membutuhkan transfer memori kumulatif sebesar $900$ GB. Pada akselerator seperti NVIDIA Tesla T4 dengan lebar pita efektif $\sim 240$ GB/s, kecepatan generasi secara fisik terhambat pada batas 25 hingga 35 token per detik. Hal ini menimbulkan latensi mutlak 4 hingga 6 detik per transaksi. Pendekatan perutean terarah berbasis tata bahasa formal (seperti Outlines [7]) mampu menjaga format sintaksis, namun tidak mengeliminasi latensi transmisi memori yang inheren pada generasi token sekuensial.

### B. Arsitektur Keputusan Non-Autoregresif
Arsitektur non-autoregresif menanggalkan proses pembangkitan kata sekuensial, menghitung representasi secara serentak pada seluruh posisi [8]. Model enkoder dwiarah modern seperti ModernBERT [9] dan DeBERTa [10] memproses konteks masukan secara paralel dan memproyeksikan vektor semantik $\mathbf{h} \in \mathbb{R}^d$ langsung ke kepala klasifikasi linier atau regresi:
$$\mathbf{z} = \mathbf{W}_h \mathbf{h} + \mathbf{b}_h$$
Pendekatan mutakhir lainnya, seperti TypeSafe JEV System One [11], mengevaluasi nilai *continuation logits* pada himpunan indeks pilihan terdefinisi dalam satu lintasan maju tunggal tanpa perulangan generasi. Dengan memformulasikan:
$$\Pr(a_k \mid X) = \frac{\exp(z_{a_k})}{\sum_{j=1}^{K} \exp(z_{a_j})}$$
model mampu menghasilkan distribusi probabilitas keputusan yang terkalibrasi penuh dalam rentang 50 ms hingga 200 ms. Karena luaran dipetakan langsung ke struktur data bertipe statis pada memori sistem peladen, model keputusan non-autoregresif menjamin kepatuhan sintaksis 100% tanpa mengonsumsi kuota token generasi.

### C. Teori Antrean dan Standar Tanggap Darurat
Penerapan teori antrean stokastik pada pusat panggilan tanggap darurat dan kantor pelayanan publik telah menjadi landasan riset manajemen operasional [12], [13]. Pada sistem panggilan darurat 112 atau 911, keterlambatan respons secara langsung meningkatkan risiko fatalitas jiwa dan kerugian aset [14]. Standar operasional internasional dan nasional—seperti NFPA 1221, EENA, dan pedoman teknis Kementerian Komunikasi dan Informatika untuk layanan panggilan darurat 112—mengharuskan proses triase awal dan pemberitahuan pos komando armada diselesaikan dalam waktu 3 hingga 15 detik. Ketika agen AI menggantikan operator manusia pada lapis pertama, karakteristik waktu layanan AI mendikte waktu tunggu antrean agregat. Literatur manajemen operasional menetapkan bahwa utilisasi peladen di atas 85% pada sistem multi-peladen Markovian akan memicu ketidakstabilan antrean yang tajam. Menempatkan model dengan waktu layanan multi-detik ke dalam kanal triase bervolume tinggi melanggar kapasitas batas kestabilan antrean.

---

## III. Formulasi Matematis dan Teori Antrean

### A. Formulasi Dinamika Waktu Tunggu Antrean Erlang-C
Pandang gerbang penerimaan laporan masyarakat sebagai sistem antrean $M/M/c$. Laporan warga masuk mengikuti proses Poisson dengan laju kedatangan agregat $\lambda$ laporan per detik. Infrastruktur pemrosesan diperkuat oleh $c$ unit peladen komputasi paralel (pekerja GPU/CPU). Masing-masing pekerja memproses aduan dengan durasi waktu layanan yang terdistribusi eksponensial independen dengan nilai rata-rata $\bar{t} = 1/\mu$, di mana $\mu$ adalah laju penyelesaian laporan per pekerja.

Beban lalu lintas yang ditawarkan pada sistem dinyatakan dalam satuan Erlang:
$$a = \frac{\lambda}{\mu} = \lambda \cdot \bar{t}$$
Tingkat utilisasi peladen $\rho$ di seluruh $c$ pekerja paralel adalah:
$$\rho = \frac{a}{c} = \frac{\lambda}{c \mu}$$
Agar sistem stabil secara ergodik, laju kedatangan tidak boleh melampaui kapasitas pemrosesan total:
$$\rho < 1 \iff \lambda < c \mu$$
Jika kondisi kestabilan terpenuhi ($\rho < 1$), probabilitas bahwa laporan warga yang baru masuk menemukan seluruh $c$ pekerja sedang sibuk dihitung melalui formula Erlang-C:
$$C(c, a) = \frac{\frac{a^c}{c!} \frac{1}{1 - \rho}}{\sum_{k=0}^{c-1} \frac{a^k}{k!} + \frac{a^c}{c!} \frac{1}{1 - \rho}}$$
Ekspektasi waktu tunggu dalam antrean $W_q$ sebelum suatu laporan dievaluasi oleh sistem AI adalah:
$$W_q = \frac{C(c, a)}{c \mu (1 - \rho)} = \frac{C(c, a)}{\lambda \left(\frac{1}{\rho} - 1\right)} = \frac{C(c, a)}{\lambda (1 - \rho)}$$
Total waktu penyelesaian sistem $T_{\text{sys}}$, yang menggabungkan waktu tunggu antrean dan durasi inferensi AI, adalah:
$$T_{\text{sys}} = W_q + \bar{t} = \frac{C(c, a)}{c \mu (1 - \rho)} + \frac{1}{\mu}$$

**Ledakan Antrean saat Lonjakan Bencana:**  
Misalkan suatu bencana hidrometeorologi besar memicu lonjakan pelaporan dengan laju kedatangan $\lambda = 50 \text{ laporan/dtk}$. Perhatikan kluster peladen yang diperkuat oleh $c = 16$ pekerja GPU paralel:
- **Kasus 1: Model Keputusan Non-Autoregresif (JEV System One):**  
  $\bar{t}_{\text{JEV}} = 0{,}1976 \text{ dtk} \implies \mu_{\text{JEV}} = \frac{1}{0{,}1976} \approx 5{,}0607 \text{ kueri/dtk}$.  
  Kapasitas total: $c \mu_{\text{JEV}} = 16 \times 5{,}0607 = 80{,}97 \text{ laporan/dtk}$.  
  Utilisasi peladen: $\rho_{\text{JEV}} = \frac{50}{80{,}97} \approx 0{,}6175 < 1{,}0$.  
  Antrean sangat stabil: $C(16; 9{,}88) \approx 0{,}054 \implies W_q^{\text{JEV}} \approx 1{,}74 \text{ ms}$.  
  Total waktu respons sistem: $T_{\text{sys}} \approx 199{,}3 \text{ ms}$.
- **Kasus 2: LLM Fondasional Autoregresif (Sahabat-AI 8B):**  
  $\bar{t}_{\text{LLM}} = 5{,}1078 \text{ dtk} \implies \mu_{\text{LLM}} = \frac{1}{5{,}1078} \approx 0{,}1958 \text{ kueri/dtk}$.  
  Kapasitas total: $c \mu_{\text{LLM}} = 16 \times 0{,}1958 = 3{,}13 \text{ laporan/dtk}$.  
  Beban yang ditawarkan: $a_{\text{LLM}} = 50 \times 5{,}1078 = 255{,}39 \text{ Erlang}$.  
  Utilisasi meledak: $\rho_{\text{LLM}} = \frac{255{,}39}{16} \approx 15{,}96 \gg 1{,}0$.  
  Antrean divergen secara deterministik:
  $$\frac{d Q(t)}{dt} = \lambda - c \mu_{\text{LLM}} = 50 - 3{,}13 = 46{,}87 \text{ laporan/dtk}$$
  Dalam 60 detik pertama, 2.812 laporan warga menumpuk dalam antrean, koneksi habis, dan sistem mengalami kelumpuhan total.

Untuk mempertahankan $\rho \le 0{,}75$ dengan laju $\lambda = 50 \text{ laporan/dtk}$ menggunakan LLM 8B, diperlukan:
$$c_{\text{diperlukan}}^{\text{LLM}} = \left\lceil \frac{255{,}39}{0{,}75} \right\rceil = 341 \text{ unit GPU}$$
Sebaliknya, model keputusan JEV hanya memerlukan:
$$c_{\text{diperlukan}}^{\text{JEV}} = \left\lceil \frac{9{,}88}{0{,}75} \right\rceil = 14 \text{ pekerja}$$
Hal ini merefleksikan penghematan infrastruktur fisik peladen sebesar 24,3 kali lipat.

### B. Dekomposisi Waktu Disposisi dan Peluruhan Resusitasi Klinis
Pada operasi panggilan darurat 112, waktu pemrosesan didekomposisi menjadi dua fase: (1) **Time-to-Dispatch** ($T_{\text{dispatch}}$), jeda waktu sejak panggilan diterima hingga perintah armada dikirim ke CAD, dan (2) **Total Flow Time** ($T_{\text{flow}}$), total durasi hingga seluruh panduan suara selesai disampaikan.

Pada LLM autoregresif mandiri, peluncuran armada terikat secara sekuensial pada pembentukan struktur JSON:
$$T_{\text{dispatch}}^{\text{LLM}} = T_{\text{flow}}^{\text{LLM}} = t_{\text{prefill}} + \sum_{k=1}^{L_{\text{JSON}}} t_{\text{decode}}^{(k)}$$
Pemberangkatan kendaraan fisik tertunda sepanjang cakrawala generasi teks ($L_{\text{JSON}} \approx 187 \text{ token}$, memakan waktu $\sim 4.984{,}1 \text{ ms}$).

Dalam kedokteran darurat, kelangsungan hidup henti jantung di luar rumah sakit (OHCA) meluruh secara eksponensial:
$$P_{\text{survival}}(t + \Delta t_{\text{dispatch}}) = P_{\text{survival}}(t) \cdot e^{-\lambda_{\text{CPR}} \Delta t_{\text{dispatch}}} \approx P_{\text{survival}}(t) \cdot (1 - \lambda_{\text{CPR}} \Delta t_{\text{dispatch}})$$
di mana $\lambda_{\text{CPR}} \approx 0{,}0017 \text{ dtk}^{-1} \text{ hingga } 0{,}0020 \text{ dtk}^{-1}$ (peluang hidup menurun 7% hingga 10% per menit tanpa pertolongan). Penundaan generasi autoregresif selama 5 detik ($T_{\text{dispatch}}^{\text{LLM}} \approx 5{,}0 \text{ dtk}$) memicu penalti kelangsungan hidup klinis secara langsung:
$$\begin{aligned}
\Delta P_{\text{survival}} &\approx - \lambda_{\text{CPR}} \cdot \Delta t_{\text{dispatch}} \\
&\approx -1{,}0\% \text{ penurunan mutlak peluang hidup}
\end{aligned}$$
Pada skala 10.000 panggilan henti jantung per tahun di kota metropolitan, keterlambatan 5 detik akibat generasi teks menyebabkan sekitar 100 kematian yang seharusnya dapat dicegah sebelum roda ambulans berputar.

Pada **Pipeline Tandem Hibrida Dua-Tingkat**, disposisi armada didekopling dari sintesis percakapan:
$$\begin{aligned}
T_{\text{dispatch}}^{\text{Tandem}} &= t_{\text{Tingkat 1}} = t_{\text{DM}} \le 200 \text{ ms} \\
T_{\text{guidance}}^{\text{Tandem}} &= t_{\text{Tingkat 1}} + t_{\text{Tingkat 2}} \approx 5{.}961{,}2 \text{ ms}
\end{aligned}$$
Armada penyelamat diberangkatkan pada $t = 196{,}2 \text{ ms}$ (memenuhi SLA $\tau_{\text{SLA}} = 3{,}0 \text{ dtk}$), sementara instruksi kompresi dada RJP disintesis secara paralel di luar jalur kritis pengiriman.

### C. Penyaringan Badai Panggilan Iseng / Hoaks saat Krisis
Saat bencana alam, pusat panggilan 112 kerap menghadapi badai panggilan, di mana 60% hingga 80% lalu lintas merupakan panggilan iseng/hoaks ($p_{\text{prank}} \in [0{,}60, 0{,}80]$).

Pada alur LLM autoregresif, setiap panggilan iseng memakan $\bar{t}_{\text{prank}} \approx 4.605{,}9 \text{ ms}$ komputasi GPU:
$$a_{\text{terbuang}}^{\text{LLM}} = p_{\text{prank}} \cdot \lambda \cdot \bar{t}_{\text{prank}}$$
Jika $p_{\text{prank}} = 0{,}70$ dan $\lambda = 30 \text{ panggilan/dtk}$:
$$a_{\text{terbuang}}^{\text{LLM}} = 0{,}70 \times 30 \times 4{,}6059 \approx 96{,}72 \text{ Erlang}$$
menuntut lebih dari 129 unit GPU komersial hanya untuk merespons penelepon iseng.

Sebaliknya, model keputusan non-autoregresif mengklasifikasikan panggilan iseng dalam satu lintasan maju ($\bar{t}_{\text{prank}}^{\text{DM}} \approx 0{,}2235 \text{ dtk}$, 0 token):
$$a_{\text{terbuang}}^{\text{DM}} = 0{,}70 \times 30 \times 0{,}2235 \approx 4{,}69 \text{ Erlang}$$
Model keputusan melepaskan 95,1% pemborosan komputasi, memutus panggilan iseng dalam 223,5 ms dan mengamankan saluran antrean untuk krisis nyata.

### D. Probabilitas Pelanggaran Batas Waktu SLA
Dengan asumsi latensi $\Delta t \sim \mathcal{N}(\mu_t, \sigma_t^2)$, probabilitas pelanggaran SLA adalah:
$$P(\text{Pelanggaran SLA}) = 1 - \Phi\left(\frac{\tau_{\text{SLA}} - \mu_t}{\sigma_t}\right) = Q\left(\frac{\tau_{\text{SLA}} - \mu_t}{\sigma_t}\right)$$
Untuk TypeSafe JEV System One ($\mu_t = 197{,}6 \text{ ms}, \sigma_t = 40{,}5 \text{ ms}$):
$$z_{\text{JEV}} = \frac{500 - 197{,}6}{40{,}5} \approx +7{,}47 \implies P(\text{Pelanggaran})_{\text{JEV}} \approx 3{,}9 \times 10^{-14} \approx 0{,}0\%$$
Untuk Sahabat-AI 8B Instruct ($\mu_t = 5.107{,}8 \text{ ms}, \sigma_t = 484{,}5 \text{ ms}$):
$$z_{\text{LLM}} = \frac{500 - 5.107{,}8}{484{,}5} \approx -9{,}51 \implies P(\text{Pelanggaran})_{\text{LLM}} = 1{,}000 \text{ (100,0\%)} \text{ (Pelanggaran Mutlak)}$$

### E. Disipasi Energi dan Total Biaya Kepemilikan (TCO)
Konsumsi daya pada peladen akselerator ganda ($P_{\text{node}} = 300 \text{ W}$):
$$\begin{aligned}
E_{\text{kueri}}^{\text{LLM}} &= 300 \text{ W} \times 5{,}1078 \text{ dtk} = 1.532{,}34 \text{ J} \approx 0{,}4256 \text{ Wh} \\
E_{\text{kueri}}^{\text{DM}} &= 70 \text{ W} \times 0{,}050 \text{ dtk} = 3{,}50 \text{ J} \approx 0{,}00097 \text{ Wh} \\
\frac{E_{\text{kueri}}^{\text{LLM}}}{E_{\text{kueri}}^{\text{DM}}} &= \frac{1.532{,}34}{3{,}50} \approx 437{,}8 \times
\end{aligned}$$
Pada tarif sewa komputasi komersial \$0,20 per $10^6$ token:
$$\text{Biaya}_{\text{100k}}^{\text{LLM}} \approx \$9{,}92 \text{ hingga } \$39{,}20$$
Model keputusan non-autoregresif mengonsumsi nol token ($N_{\text{tok}} \equiv 0$), dengan biaya panggilan logit \$0,05 per 100.000 kueri:
$$\eta_{\text{biaya}} = \left( 1 - \frac{\$0{,}05}{\$39{,}20} \right) \times 100\% = 99{,}872\%$$
Model keputusan memangkas 99,87% biaya komputasi berulang.

### F. Hukum Entropi Skema dan Kegagalan Sintaksis JSON
Pada dekoder autoregresif yang menyintesis teks sepanjang $L$ token dengan tingkat pergeseran token $\bar{p}_{\text{drift}}$:
$$P(\text{Gagal Skema}) = 1 - \prod_{k=1}^{L} (1 - p_{\text{drift}, k}) \approx 1 - (1 - \bar{p}_{\text{drift}})^L$$
Untuk $\bar{p}_{\text{drift}} \approx 0{,}00181$ pada panjang $L = 196$ token:
$$P(\text{Gagal Skema}) \approx 1 - (1 - 0{,}00181)^{196} \approx 1 - 0{,}7015 = 0{,}2985 \approx 30{,}0\%$$
Persamaan ini membuktikan secara analitis mengapa Sahabat-AI 8B mencatat tingkat kegagalan parsing skema sebesar 30,0%. Pada model non-autoregresif, $p_{\text{drift}, k} \equiv 0$, sehingga $P(\text{Kepatuhan Skema}) \equiv 100{,}0\%$.

---

## IV. Metodologi Tolok Ukur Empiris

### A. Lingkungan Pengujian dan Topologi Komputasi
Tolok ukur diimplementasikan dalam repositori **DecisionModelBench** pada peladen multi-GPU terdistribusi yang diperkuat oleh dua kartu akselerator NVIDIA Tesla T4 (masing-masing 16 GB VRAM, total 31,27 GB VRAM, PCIe Gen3 x16, lebar pita $300$ GB/s per kartu). Peladen menggunakan OS Ubuntu 22.04 LTS ditenagai prosesor Intel Xeon (2,20 GHz, 4 vCPU) dan RAM sistem 32 GB.
- **GPU 0 (Lapis Generatif & Enkoder Lokal):** Menginangi LLM Sahabat-AI 8B Instruct (GGUF Q4_K_M, menyerap $\sim 5{,}2$ GB VRAM) serta kepala keputusan lokal Kev-0,8B dan Laya Multilingual 421M.
- **GPU 1 (OpenJev Logit Scorer):** Menginangi OpenJev 0,5B yang didedikasikan untuk penilaian logit kandidat satu-lintasan.
- **Tingkat SaaS API (TypeSafe JEV System One):** Dihubungkan melalui REST API aman melewati jaringan WAN berkecepatan tinggi.

### B. Model yang Dievaluasi
1. **TypeSafe JEV System One:** Model keputusan awan berbasis ekstraksi logit satu-lintasan tanpa generasi token yang menghasilkan flag boolean terstruktur, target pilihan, dan skor urgensi kontinu.
2. **OpenJev (0,5B Logit Scorer):** Tulang punggung kausal 500 juta parameter lokal yang mengevaluasi nilai probabilitas logit pada pilihan terdefinisi dalam satu lintasan maju.
3. **Kev-0,8B (LoRA Pointer):** Model 800 juta parameter lokal dengan kepala penunjuk LoRA untuk perutean diskret ke entitas tujuan.
4. **Laya Multilingual (421M):** Model enkoder ModernBERT 421 juta parameter dwibahasa (Indonesia-Inggris) untuk klasifikasi semantik cepat.
5. **Sahabat-AI 8B Instruct:** Model Bahasa Besar fondasional kedaulatan Indonesia berparameter 8,0 miliar yang dikembangkan oleh GoTo dan Indosat Ooredoo Hutchison, diinstruksikan khusus untuk domain administratif Indonesia [3].
6. **Qwen 2.5 7B Instruct:** Model Bahasa Besar fondasional 7,6 miliar parameter yang dikembangkan oleh Alibaba Cloud [4], dioptimalkan untuk penalaran multibahasa, pemrograman, dan luaran terstruktur JSON.
7. **Gemma 2 9B Instruct:** Model fondasional bobot-terbuka 9,2 miliar parameter yang dikembangkan oleh Google DeepMind [16], yang memadukan arsitektur atensi *sliding-window* lokal dan atensi global.

### C. Skenario Administrasi Publik dan Layanan Korporat
Sepuluh skenario dunia nyata berisiko tinggi diuji:
1. **PUB-01 (Evakuasi Banjir Bandang):** Tanggul Sungai Ciliwung RW 07 jebol, merendam permukiman 2 meter dengan lansia dan bayi terjebak. Target: `bpbd_basarnas_damkar`.
2. **PUB-02 (Pungli Pembuatan e-KTP / SP4N-LAPOR):** Laporan pungutan liar Rp 150.000 untuk mempercepat e-KTP di kelurahan. Target: `inspektorat_ombudsman`.
3. **PUB-03 (Amblasnya Jalan Nasional Trans-Provinsi):** Lubang amblas 1,5 meter akibat gerusan hujan menggulingkan truk tronton dan memicu kemacetan 10 km. Target: `pupr_bina_marga`.
4. **PUB-04 (Penolakan Pasien Darurat di RS):** IGD RS swasta menolak pasien serangan jantung akut karena alasan kamar penuh dan meminta uang muka Rp 20.000.000 meskipun berstatus peserta aktif BPJS Kesehatan. Target: `kemenkes_bpjs_kesehatan`.
5. **PUB-05 (Pembuangan Limbah B3 Beracun):** Pabrik tekstil membuang limbah kimia berbahaya ke sungai pada pukul 02:00 dini hari. Target: `gakkum_klhk_dlh`.
6. **CS-01 (Pembobolan Rekening Perbankan / OTP Phishing):** Pengaduan pengambilalihan akun dan pengurasan saldo tabungan Rp 25.000.000 via telepon penipuan. Target: `fraud_desk_emergency_block`.
7. **CS-02 (Sengketa E-Commerce / Ancaman Viral):** Pembelian ponsel pintar diganti sabun colek dalam paket pengiriman dan pembeli mengancam memviralkan di media sosial. Target: `priority_dispute_retur`.
8. **CS-03 (Restrukturisasi Kredit Korban PHK):** Nasabah ter-PHK akibat pailit mengajukan perpanjangan tenor cicilan secara beriktikad baik. Target: `collection_restructuring`.
9. **CS-04 (Pemutusan Tulang Punggung Serat Optik):** Alat berat ekskavator memutus jalur utama serat optik kawasan industri. Target: `noc_fiber_splicing_team`.
10. **CS-05 (Serangan Siber Prompt Injection):** Perintah manipulasi yang mencoba membobol instruksi sistem guna mengekstrak kredensial basis data dan data pribadi warga. Target: `security_block_and_log`.

### D. Skenario Panggilan Darurat 112/911 Penyelamatan Jiwa
Lima skenario kritis panggilan darurat 112 diformulasikan:
1. **EMERG-01: Henti Jantung & Henti Napas (Cardiac Arrest & Apnea):** Penelepon histeris melaporkan ayahnya roboh memegangi dada, tidak sadar, napas terhenti, dan wajah membiru di Jalan Anggrek No. 12. Nilai Acuan: Darurat: `True`; Target: `ambulans_paramedis_icu`; Kode Triase: `merah_kritis_henti_jantung`. Tindakan: Peluncuran ambulans paramedis ICU instan disertai panduan suara kompresi dada RJP.
2. **EMERG-02: Kebakaran Gedung Ruko Terjebak Asap (Structural Fire):** Gudang plastik lantai 1 meledak, api membesar ke lantai 3, enam karyawan terjebak di *rooftop* karena tangga tertutup asap hitam panas. Nilai Acuan: Darurat: `True`; Target: `damkar_snorkel_rescue`; Kode Triase: `kritis_korban_terjebak`. Tindakan: Peluncuran mobil damkar tangga tinggi (snorkel) disertai panduan bertahan hidup dari asap.
3. **EMERG-03: Perampokan Bersenjata Berlangsung (Armed Robbery):** Penelepon berbisik dari ruang brankas minimarket di Jalan Raya Bogor KM 28, melaporkan tiga pelaku bersenjata tajam dan pistol menyekap kasir. Nilai Acuan: Darurat: `True`; Target: `polri_resmob_gegana`; Kode Triase: `ancaman_senjata_mematikan`. Tindakan: Peluncuran tim taktis Resmob/Perintis Polri senyap tanpa sirine disertai panduan bersembunyi.
4. **EMERG-04: Tabrakan Beruntun Tol Cipularang Korban Terjepit (Extrication Crash):** Truk tronton rem blong menabrak 5 kendaraan di Tol Cipularang KM 92, dua korban terjepit kabin ringsek berdarah-darah. Nilai Acuan: Darurat: `True`; Target: `basarnas_rescue_hidrolik`; Kode Triase: `kritis_terjepit_nyawa`. Tindakan: Pemberangkatan tim penyelamat Basarnas/Damkar dengan alat pemotong hidrolik (*Jaws of Life*).
5. **EMERG-05: Panggilan Iseng / Hoaks (Prank Call Filter):** Penelepon iseng menanyakan jam dan menanyakan apakah operator 112 menjual pulsa telepon seluler. Nilai Acuan: Darurat: `False`; Target: `filter_prank_warning`; Kode Triase: `prank_palsu`. Tindakan: Peringatan tegas dan pemutusan saluran seketika tanpa menyerap komputasi GPU.

### E. Arsitektur Pipeline Tandem Hibrida Dua-Tingkat
Pipeline Tandem Hibrida mengevaluasi sinergi model keputusan dan LLM fondasional yang bekerja secara paralel pada peladen Dual Tesla T4:
- **Tingkat 1 (Gerbang Refleks Keputusan Sub-200ms):** Menerima transkrip aduan warga dan mengevaluasi logit kandidat dalam satu lintasan maju. Tingkat 1 menghasilkan tag terstruktur (`is_emergency`, `dispatch_unit`, `triage_code`) dalam waktu di bawah 200 ms, menerbitkan perintah peluncuran armada langsung ke sistem CAD. Tingkat 1 tidak menghasilkan satu token teks pun ($N_{\text{tok}} \equiv 0$).
- **Tingkat 2 (Mesin Bimbingan Fondasional Asinkron):** Berjalan di GPU 0 menggunakan Sahabat-AI 8B Instruct. Tingkat 2 menerima transkrip dan tag triase terstruktur dari Tingkat 1. Berjalan di luar jalur kritis pengiriman armada (asinkron), Tingkat 2 menyintesis instruksi pertolongan pertama (seperti ketukan irama kompresi RJP atau panduan evakuasi asap) kepada penelepon dalam 3 hingga 7 detik tanpa menunda keberangkatan kendaraan fisik.

---

## V. Hasil Empiris dan Pembahasan Kinerja

### Tabel I: Evaluasi Empiris Administrasi Publik dan Layanan Korporat
*Evaluasi Empiris Komprehensif: Model Keputusan Non-Autoregresif Melawan LLM Generatif Fondasional pada Administrasi Publik dan Layanan Korporat Bervolume Tinggi (DecisionModelBench).*

| Arsitektur Model | Params | Paradigma | Latensi Rata-rata (ms) | Rentang Latensi (ms) | Akurasi Perutean | Kepatuhan Skema | Rata-rata Token | Biaya / 100k Kueri | Lolos SLA ($<500$ ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TypeSafe JEV System One** | — | SaaS Cloud API | **197,6** | 144,2 – 294,2 | **100,0%** | **100,0%** | **0** | **\$0,05** | **100,0%** |
| **Kev-0,8B (LoRA Pointer)** | 0,8B | GPU Lokal | 648,5 | 220,9 – 1.182,0 | **100,0%** | **100,0%** | **0** | **\$0,05** | 20,0% |
| **OpenJev (Logit Scorer)** | 0,5B | GPU Lokal | 540,6 | 499,6 – 582,3 | 90,0% | **100,0%** | **0** | **\$0,05** | 10,0% |
| **Laya Multilingual** | 421M | GPU/CPU Lokal | 3.639,2* | 63,2 – 7.211,7 | 60,0% | **100,0%** | **0** | **\$0,05** | 10,0% |
| **Sahabat-AI 8B Instruct** | 8,0B | LLM Autoregresif | 5.107,8 | 4.305,8 – 5.703,2 | 70,0% | 70,0% | 196 | \$39,20 | **0,0%** |

*\*Catatan: Laya Multilingual menyelesaikan lintasan maju dalam 35,8–63,2 ms pada lingkungan komputasi terisolasi (misal 63,2 ms pada skenario PUB-02).*

---

### Tabel II: Evaluasi Empiris Panggilan Darurat (911 / 112) dan Pipeline Tandem Hibrida
*Evaluasi Empiris Panggilan Darurat (911 / 112) dan Pipeline Tandem Hibrida: Model Keputusan Mandiri Melawan LLM Fondasional Mandiri Melawan Pipeline Tandem Hibrida pada Akselerator Ganda NVIDIA Tesla T4.*

| Paradigma Operasional | Time-to-Dispatch ($T_{\text{dispatch}}$) | Total Flow Time ($T_{\text{flow}}$) | Akurasi Perutean | Emisi Token | Status Regulasi ($<3{,}0$ dtk SLA) | Penilaian Klinis / Operasional |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model Keputusan (JEV Cloud)** | **232,8 ms** | **232,8 ms** | **100,0%** (5/5) | **0** | **Lolos Regulasi** | Pengiriman Armada Seketika |
| **Model Keputusan (OpenJev 0,5B)** | 560,4 ms | 560,4 ms | 80,0% (4/5) | **0** | **Lolos Regulasi** | Eksekusi Lokal Deterministik |
| **LLM Mandiri (Sahabat-AI 8B)** | 4.984,1 ms | 4.984,1 ms | **100,0%** (5/5) | 187 | **GAGAL SLA** (Tunda) | Bahaya Jiwa (Krisis Periode Emas) |
| **TWO-TIER HYBRID TANDEM** | **196,2 ms** | 5.961,2 ms | **100,0%** (5/5) | 226 | **PARETO OPTIMAL** | **Armada Instan + Bimbingan RJP** |

---

### A. Distribusi Latensi dan Kepatuhan SLA pada Triase Sipil
Tabel I membuktikan keunggulan mutlak model keputusan non-autoregresif atas LLM fondasional. **TypeSafe JEV System One** mencatat latensi rata-rata **197,6 ms** dengan **100,0% tingkat kelulusan SLA** ($<500$ ms). Model lokal **OpenJev 0,5B** mencatat latensi stabil **540,6 ms** ($\sigma = 24{,}3$ ms) dan **Kev-0,8B** mencatat **648,5 ms**.

Sebaliknya, **Sahabat-AI 8B Instruct** mencatat latensi rata-rata **5.107,8 ms** (25 kali lebih lambat dibandingkan JEV System One), mencatatkan **tingkat kelulusan SLA tepat 0,0%**.

### B. Akurasi Perutean Semantik
**TypeSafe JEV System One** dan **Kev-0,8B** meraih **100,0% akurasi perutean** pada 10 skenario sipil dan korporat. **OpenJev 0,5B** meraih **90,0% akurasi**. **Sahabat-AI 8B Instruct** hanya mencapai **70,0% akurasi**, mengalami kegagalan pada PUB-04 (Penolakan Pasien BPJS), CS-03 (Restrukturisasi Utang), dan CS-04 (Putus Kabel Serat Optik).

### C. Kepatuhan Skema dan Kegagalan Parsing JSON
Model keputusan non-autoregresif meraih **100,0% kepatuhan skema** tanpa eksepsi parsing sintaksis. Sebaliknya, Sahabat-AI 8B mencatat **30,0% kegagalan parsing skema JSON** (70,0% kepatuhan), memunculkan kesalahan pembatas koma:
```
JSON parse error: Expecting ',' delimiter: line 17 column 4 (char 318)
```
dan
```
JSON parse error: Expecting ',' delimiter: line 16 column 4 (char 292)
```

### D. Patologi Kontensi Komputasi Peladen
Pada lingkungan terisolasi, **Laya Multilingual 421M** mengeksekusi inferensi dalam **35,8 ms hingga 63,2 ms** (63,2 ms pada PUB-02). Namun, saat dijalankan bersamaan dengan proses LLM 8B pada perangkat keras yang tidak terpartisi, perebutan memori dan utas CPU menggelembungkan latensi Laya menjadi **3.639,2 ms**, membuktikan perlunya isolasi fisik akselerator untuk model keputusan.

### E. Evaluasi Empiris Panggilan Darurat (911 / 112) dan Pengujian Tandem
Tabel II menyajikan perbandingan empiris lintas paradigma pada lima skenario kritis penyelamatan nyawa:
1. **Dekomposisi Time-to-Dispatch:** LLM Mandiri Sahabat-AI 8B membutuhkan rata-rata *Time-to-Dispatch* **4.984,1 ms** ($T_{\text{dispatch}}$), melanggar standar batas waktu darurat 112 pada 100% pengujian. Sebaliknya, **Pipeline Tandem Hibrida Dua-Tingkat** mencatat *Time-to-Dispatch* instan **196,2 ms** (0 token), memberangkatkan kendaraan penyelamat dalam seperlima detik. Secara paralel, Tingkat 2 menyelesaikan bimbingan verbal dalam 5.765,0 ms ($T_{\text{flow}} = 5.961{,}2 \text{ ms}$), menghasilkan 226 token panduan penyelamatan jiwa yang mendalam dan empatik.
2. **Wawasan Medis: Penyelamatan Periode Emas Pasien Henti Jantung:** Pada EMERG-01 (Henti Jantung & Henti Napas), LLM mandiri menahan keberangkatan ambulans selama **5.841,7 ms** saat menyintesis teks deskriptif. Pada Pipeline Tandem Hibrida, sinyal CAD diterbitkan pada **$t = 217{,}0$ ms**, sementara Tingkat 2 membimbing penelepon melakukan kompresi dada RJP dengan irama 100–120 kali per menit. Hal ini mengamankan detik-detik berharga periode emas (3–5 menit).
3. **Analisis Tanggap Bencana: Penyaringan Badai Panggilan Iseng:** Pada EMERG-05 (Panggilan Iseng), Sahabat-AI 8B menghabiskan **4.605,9 ms** dan **171 token** komputasi GPU untuk menolak permintaan pulsa. Sebaliknya, Model Keputusan (JEV) mengklasifikasikan panggilan iseng dalam **223,5 ms** tanpa emisi token, memangkas 95,1% beban komputasi dan membebaskan antrean untuk panggilan darurat nyata.

---

### Tabel III: Evaluasi Empiris Multi-LLM Tandem Lintas Arsitektur Unggulan
*Evaluasi Empiris Multi-LLM Tandem Lintas Arsitektur Unggulan: LLM Mandiri Satu-Tingkat Melawan Pipeline Tandem Hibrida Dua-Tingkat pada Akselerator Ganda NVIDIA Tesla T4 (DecisionModelBench).*

| Model Arsitektur (Mesin Sistem 2) | Parameter & Pengembang | Latensi Solo (ms) | Disp. Tandem ($T_{\text{disp}}$) | Aksel. Disp. | Akurasi Solo | Akurasi Tandem | Panduan P3K (Token RJP) | Total Aliran ($T_{\text{flow}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TypeSafe JEV (Sys 1 Mandiri)** | Non-AR / TypeSafe AI | **208,2 ms** | **208,2 ms** | 1,0$\times$ | **100,0%** | **100,0%** | 0 token | **208,2 ms** |
| **Sahabat-AI 8B Instruct** | 8,0B / Indosat-GoTo | 3.340,0 ms | 217,7 ms | 15,3$\times$ | **100,0%** | **100,0%** | 294 token | 7.452,0 ms |
| **Qwen 2.5 7B Instruct** | 7,6B / Alibaba Cloud | 10.024,4 ms | **178,5 ms** | **56,1$\times$** | 33,3%* | **100,0%** | 281 token | 6.595,3 ms |
| **Gemma 2 9B Instruct** | 9,2B / Google DeepMind | 12.681,7 ms | 197,4 ms | **64,2$\times$** | **100,0%** | **100,0%** | 207 token | 6.909,0 ms |

*\*Catatan: Qwen 2.5 7B Mandiri mengalami kejatuhan akurasi ke 33,3% akibat pembungkusan luaran JSON dalam pagar kode markdown (` ```json ... ``` `), yang memicu eksepsi parsing sintaksis fatal pada mesin CAD satu-tingkat. Pada arsitektur Tandem Hibrida Dua-Tingkat, TypeSafe JEV Sistem 1 mengisolasi lapisan CAD secara deterministik, mengeksekusi pengiriman armada dalam 178,5 ms (akurasi 100,0%) sembari membiarkan Qwen 2.5 menyintesis 281 token panduan pertolongan pertama secara asinkron di luar jalur kritis.*

---

### F. Evaluasi Tandem Multi-LLM Lintas Arsitektur Unggulan (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B)
Guna menguji apakah keunggulan operasional Pipeline Tandem Hibrida Dua-Tingkat berlaku secara konsisten melintasi berbagai keluarga model bahasa besar terkemuka, kami melakukan uji tegangan empiris pada peladen komputasi ganda NVIDIA Tesla T4 terhadap tiga arsitektur bobot-terbuka unggulan dunia:
1. **Sahabat-AI 8B Instruct** (Indosat/GoTo, 8,0 miliar parameter);
2. **Qwen 2.5 7B Instruct** (Alibaba Cloud, 7,6 miliar parameter);
3. **Gemma 2 9B Instruct** (Google DeepMind, 9,2 miliar parameter).

Tabel III merangkum perbandingan empiris antara eksekusi LLM mandiri satu-tingkat (*single-tier*) melawan Pipeline Tandem Hibrida Dua-Tingkat (bermitra dengan TypeSafe JEV System 1). Data empiris ini menunjukkan jurang operasional yang sangat masif:
- **Latensi Acuan Mandiri:** Pada mode mandiri, model keputusan non-autoregresif acuan (TypeSafe JEV) menuntaskan penentuan disposisi dalam **208,2 ms** dengan akurasi 100,0% dan nol token. Sebaliknya, LLM generatif mengalami pembengkakan waktu yang ekstrem: Sahabat-AI 8B membutuhkan waktu rata-rata **3.340,0 ms**, Qwen 2.5 7B membutuhkan **10.024,4 ms** (10 detik), dan Gemma 2 9B menelan waktu **12.681,7 ms** (12,7 detik) per panggilan.
- **Faktor Akselerasi Pengiriman Armada pada Sistem Tandem:** Saat dipadukan dengan TypeSafe JEV pada Pipeline Tandem Hibrida Dua-Tingkat, latensi pengiriman fisik armada CAD anjlok ke ranah sub-220ms pada seluruh model: **217,7 ms** untuk Sahabat-AI (percepatan **15,3$\times$**), **178,5 ms** untuk Qwen 2.5 (percepatan **56,1$\times$**), dan **197,4 ms** untuk Gemma 2 (percepatan **64,2$\times$**).
- **Ketahanan Skema dan Kerapuhan Format Sintaksis:** Bila Sahabat-AI 8B dan Gemma 2 9B mempertahankan akurasi 100,0% pada mode mandiri di skenario darurat ini, Qwen 2.5 7B anjlok ke **33,3% akurasi mandiri**. Kegagalan fatal ini berpangkal pada format struktural: Qwen membungkus luaran JSON ke dalam tanda markdown (` ```json ... ``` `) serta menambahkan teks pengantar percakapan. Pada arsitektur satu-tingkat, non-determinisme sintaksis ini mematahkan pembaca data JSON otomatis CAD, menggagalkan peluncuran armada. Sebaliknya, dalam konfigurasi Tandem Dua-Tingkat, Tingkat 1 mengambil alih penentuan disposisi secara deterministik (meraih **akurasi 100,0%** pada seluruh model), membentengi sistem operasional darurat dari variasi sintaksis LLM.
- **Bimbingan Medis dan P3K Berkualitas Tinggi:** Secara simultan di luar jalur kritis, model Tingkat 2 berhasil menyintesis panduan pertolongan pertama yang mendalam: Sahabat-AI menghasilkan rata-rata **294 token**, Qwen 2.5 menyintesis **281 token**, dan Gemma 2 menghasilkan **207 token** bimbingan RJP dan evakuasi tanpa menahan laju roda kendaraan penyelamat.

### G. Pembahasan Mendalam Sistem dan Mortalitas Klinis: Fatalitas Dekoder Single-Tier Melawan Keniscayaan Paradigma Tandem Hibrida
Hasil empiris pada Tabel III mengungkap fakta operasional yang sangat mendesak bagi keselamatan nyawa warga: **memaksa Model Bahasa Besar generatif bertindak sebagai operator disposisi tunggal (*single-tier dispatcher*) pada layanan darurat 112/911 adalah tindakan fatal yang berisiko merenggut nyawa korban di lapangan**.

#### 1. Patofisiologi Periode Emas pada Henti Jantung di Luar Rumah Sakit
Dalam kedokteran gawat darurat, prognosis pasien henti jantung di luar rumah sakit (*Out-of-Hospital Cardiac Arrest*/OHCA) ditentukan oleh hitungan detik biologis yang tidak dapat dinegosiasikan. Saat terjadi fibrilasi ventrikel atau asistol, sirkulasi darah sistemik terhenti total. Korteks serebral manusia, yang praktis tidak memiliki cadangan glikogen, kehabisan oksigen terlarut dalam 10 hingga 15 detik. Kematian sel neuron permanen (*irreversible neuronal necrosis*) dan cedera otak iskemik dimulai dalam rentang 180 hingga 300 detik (3 hingga 5 menit)—yang dikenal secara universal sebagai "periode emas" (*golden period*) resusitasi [15].

Kurva kelangsungan hidup klinis yang diformulasikan oleh Eisenberg dan Mengert [15] serta divalidasi pada triase kota modern [14] menetapkan bahwa setiap keterlambatan 60 detik dalam memulai kompresi resusitasi jantung paru (RJP) dan defibrilasi menurunkan peluang hidup sebesar 7% hingga 10%:
$$P_{\text{survival}}(t) = P_0 \cdot \exp\left(-\lambda_{\text{CPR}} \cdot t\right)$$
dengan konstanta bahaya mortalitas $\lambda_{\text{CPR}} \approx 0{,}0018 \text{ dtk}^{-1}$.

Mari kita telaah konsekuensi medis nyata jika Gemma 2 9B atau Qwen 2.5 7B dipaksa menjadi dispatcher satu-tingkat:
- **Latensi Ekstrem Gemma 2 9B (12,7 Detik Penundaan Murni):** Pembentukan muatan JSON disposisi membutuhkan waktu inferensi 12.681,7 ms pada akselerator GPU. Selama 12,7 detik tersebut, penelepon berada dalam keheningan saluran telepon, sistem CAD pusat tidak dapat menerbitkan perintah jalan, dan ambulans ICU berpemanas tetap terparkir diam di garasi posko. Keterlambatan 12,7 detik ini secara langsung memotong probabilitas kelangsungan hidup pasien:
$$\begin{aligned}
\Delta P_{\text{survival}} &= P_0 \left(1 - e^{-0{,}0018 \times 12{,}68}\right) \\
&\approx 0{,}0226 \cdot P_0 \quad (\Delta P \approx -2{,}3\%)
\end{aligned}$$
yakni reduksi kelangsungan hidup absolut sebesar 2,3%. Pada skala yurisdiksi metropolitan dengan 10.000 panggilan henti jantung per tahun, kemacetan autoregresif selama 12,7 detik ini secara langsung bertanggung jawab atas lebih dari 230 kematian yang dapat dicegah sebelum roda ambulans pertama sempat berputar.
- **Bencana Ganda Qwen 2.5 7B (10,0 Detik + Kegagalan Format 33,3%):** Selain menunda respons selama 10.024,4 ms, kegagalan akurasi mandiri Qwen 2.5 yang anjlok ke 33,3% menciptakan petaka sistemik. Pada dua dari tiga panggilan darurat, pembungkusan tanda pagar ` ```json ` memicu galat eksepsi pada sistem integrasi webhook CAD. Pada sistem operasional nyata, galat parsing ini memicu siklus pengulangan kueri (*retry*) otomatis yang menambah waktu tunda 10 hingga 20 detik, atau memaksa eskalasi manual ke operator manusia yang memakan waktu 30 hingga 45 detik. Pada kasus henti jantung atau pendarahan arteri masif, kegagalan perangkat lunak selama 30 detik merupakan vonis kematian bagi korban.

#### 2. Keniscayaan Paradigma Tandem Hibrida Dua-Tingkat
Pipeline Tandem Hibrida Dua-Tingkat secara tuntas menghapus pertentangan palsu antara kecepatan refleks operasional dan kedalaman kecerdasan linguistik. Dengan memisahkan alur pemrosesan darurat ke dalam dua strata kognitif yang terspesialisasi:
1. **Sistem 1 (Refleks Operasional Deterministik):** Model keputusan non-autoregresif (seperti TypeSafe JEV System One) mengevaluasi logit kandidat dalam satu lintasan maju. Dalam waktu **178,5 ms hingga 217,7 ms** ($<$220 ms), Sistem 1 menerbitkan direktif kategori bertipe langsung ke *event broker* CAD kota. Roda kendaraan penyelamat berputar, sirine meraung, dan sinyal prioritas lampu lalu lintas diaktifkan dalam seperlima detik—memenuhi regulasi batas waktu pengiriman armada ($\tau_{\text{dispatch}} < 3{,}0$ dtk) dengan kepatuhan skema 100% dan tanpa membuang kuota token.
2. **Sistem 2 (Musyawarah Berdaulat Asinkron):** Berjalan di luar jalur kritis pengiriman fisik melalui antrean pesan asinkron (Kafka/RabbitMQ), LLM fondasional (Gemma 2 9B, Qwen 2.5 7B, atau Sahabat-AI 8B) menyerap transkrip penelepon dan tag triase Tingkat 1. Selama ambulans melaju di jalan raya, Sistem 2 mengalirkan 200 hingga 300 token bimbingan pertolongan pertama (seperti panduan letak tumit tangan pada tulang dada, ketukan irama 100–120 kali per menit untuk kompresi RJP, teknik isolasi asap kebakaran, dan pembebasan jalan napas) langsung ke penyuara telinga operator atau layar ponsel pelapor.

Dengan demikian, arsitektur tandem menjamin bahwa kerentanan format sintaksis maupun lamanya waktu generasi pada Sistem 2 sama sekali tidak menghambat peluncuran fisik armada darurat. Faktor percepatan pengiriman mencapai **64,2$\times$**, sementara rasa tenang dan bimbingan keselamatan warga terpenuhi secara optimal. Kami menegaskan bahwa penggunaan LLM generatif mandiri satu-tingkat pada penanganan panggilan darurat penyelamatan nyawa harus dilarang secara regulasi, dan arsitektur Tandem Hibrida Dua-Tingkat wajib dijadikan standar rancang bangun infrastruktur AI publik nasional.

---

## VI. Kedaulatan Data, Tata Kelola PII, dan Kepatuhan Regulasi

### A. Kerapuhan Jendela Atensi Autoregresif terhadap PII Warga
Berdasarkan Undang-Undang Pelindungan Data Pribadi (UU PDP No. 27 Tahun 2022), pengendali data wajib menjamin kerahasiaan dan kedaulatan data perseorangan. Memasukkan narasi warga ke dalam LLM autoregresif menyimpan NIK 16-digit, nomor rekening, dan rekam medis langsung ke memori atensi *Key-Value* (KV):
$$\mathbf{K}_i = \mathbf{W}_K \mathbf{x}_i, \quad \mathbf{V}_i = \mathbf{W}_V \mathbf{x}_i$$
Hal ini menimbulkan bahaya residu memori VRAM yang tidak terenkripsi, potensi kebocoran data saat menggunakan API awan luar negeri, serta risiko halusinasi generatif yang membocorkan data warga lain.

### B. Benteng Semantik Deterministik Non-Autoregresif
Model keputusan non-autoregresif menyelesaikan risiko privasi ini secara terstruktur:
1. **Aktivasi Efemeral dan Nol Generasi:** Teks diproyeksikan ke representasi laten $\mathbf{h} \in \mathbb{R}^d$, nilai logit dihitung, dan memori segera dibersihkan. Tidak ada rekonstruksi teks dan tidak ada memori KV cache yang disimpan.
2. **Ruang Luaran yang Terbatas:** Luaran model hanya berupa indeks kategori terdefinisi $\mathcal{K}$ dan tensor angka $\mathbf{s} \in [0, 1]$, sehingga model secara matematis mustahil membocorkan data PII warga.
3. **Penerapan Tepi Berdaulat:** Kebutuhan memori yang sangat kecil ($<1$ GB VRAM) memungkinkan model dioperasikan secara lokal pada peladen pemerintah daerah yang terisolasi (*air-gapped*).

### C. Kekebalan Mutlak terhadap Manipulasi Prompt Injection
Pada skenario **CS-05**, serangan manipulasi instruksi (*prompt injection jailbreak*) mencoba membobol instruksi sistem guna mengekstrak kredensial basis data dan data pribadi warga. Model keputusan terbukti kebal secara mutlak. Karena ruang aksi dibatasi pada himpunan $\mathcal{K}$, perintah berbahaya tidak dapat mengubah alur eksekusi logika. Model mengenali niat permusuhan, memberikan nilai urgensi 4,0, dan memblokir kueri ke `security_block_and_log` dalam 179,6 ms.

---

## VII. Arsitektur Triase Sipil Dua-Tingkat Berdaulat

```
LALU LINTAS WARGA & PELANGGAN MASUK
(Panggilan 112/911, SP4N-LAPOR, Aduan Fraud Perbankan, Bencana)
         │
         ▼  [Aliran Waktu-Nyata]
┌─────────────────────────────────────────────────────────────────┐
│ TINGKAT 1: GERBANG PENAPIS NIAT (Sistem 1 - Mesin Refleks)      │
│ Model Keputusan Non-Autoregresif (JEV, OpenJev, Laya, Kev)      │
│ • Latensi Eksekusi: 196,2 ms (Deterministik, Sub-200ms)         │
│ • Luaran: Tensor Bertipe Statis, Aksi Disposisi, Urgensi        │
│ • Kepatuhan Skema: 100,0% Terjamin | Pemborosan Token: 0        │
│ • Keamanan: Perlindungan Mutlak PII & Kebal Prompt Injection    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼ [80%–90%]                     ▼ [10%–20%]
       Triase Otomatis Cepat               Penalaran Kompleks
       • Disposisi CAD Instan              • Samarkan PII via Proksi
       • Peluncuran Sirine Armada          • Antrekan ke Penengah Pesan
       • Blokir Rekening Fraud             • Picu LLM Fondasional
       • Saring Panggilan Iseng (223 ms)   • Bimbingan RJP & Empati
                                                 │
                                                 ▼ [Antrean Penengah Asinkron]
┌─────────────────────────────────────────────────────────────────┐
│ TINGKAT 2: MESIN PENALARAN BERDAULAT (Sistem 2)                 │
│ LLM Fondasional Terkuantisasi (Sahabat-AI 8B pada Dual T4)      │
│ • Latensi Eksekusi: 3 – 7 detik (Asinkron / Non-Blocking)       │
│ • Peran: Bimbingan langkah kompresi dada RJP, penenang korban   │
│   trauma, penyelesaian sengketa sela instansi, empati krisis.   │
└─────────────────────────────────────────────────────────────────┘
```

Arsitektur ini membagi pemrosesan kognitif menjadi dua lapis yang terdekopling:
1. **Tingkat 1: Gerbang Penapis Niat Berdaulat (Sistem 1 - Mesin Refleks):** Berjalan lokal pada peladen pemerintah daerah, mencegat 100% lalu lintas masuk dalam waktu di bawah 200 ms untuk menentukan status darurat, memberangkatkan armada penyelamat fisik, dan membuang panggilan iseng tanpa menyentuh LLM autoregresif.
2. **Tingkat 2: Mesin Penalaran Berdaulat (Sistem 2 - Nalar Generatif):** LLM fondasional terkuantisasi (Sahabat-AI 8B Instruct) yang diisolasi di balik sistem antrean asinkron (Kafka/RabbitMQ), khusus menangani 10% hingga 20% kasus rumit atau memberikan bimbingan suara pertolongan pertama kepada keluarga korban.

Dengan menyaring 85% lalu lintas pada Tingkat 1, kebutuhan unit GPU untuk LLM fondasional saat krisis ($\lambda = 50 \text{ laporan/dtk}$) terpangkas dari 341 GPU menjadi 52 GPU, sementara Tingkat 1 menangani 42,5 laporan per detik sisanya hanya dengan 12 pekerja kompak—menghemat lebih dari 80% biaya pengadaan infrastruktur server sambil menjaga stabilitas antrean mutlak.

---

## VIII. Kesimpulan
Penerapan langsung Model Bahasa Besar autoregresif sebagai penapis utama pada panggilan darurat 112 dan kanal pelayanan publik bervolume tinggi merupakan kekeliruan arsitektural yang fatal. Sintesis token sekuensial menimbulkan latensi multi-detik (3,3 detik hingga 12,7 detik melintasi Sahabat-AI 8B, Qwen 2.5 7B, dan Gemma 2 9B) yang memangkas secara drastis batas waktu periode emas henti jantung (3–5 menit), melanggar regulasi batas waktu pengiriman armada, serta berisiko fatal bagi keselamatan korban di lapangan. Terlebih lagi, kerapuhan sintaksis non-deterministik (seperti yang dialami Qwen 2.5 dengan akurasi mandiri 33,3% akibat pembungkusan tanda markdown) mengekspos sistem *Computer-Aided Dispatch* (CAD) pada kegagalan total yang tidak dapat ditoleransi.

Tolok ukur empiris **DecisionModelBench** membuktikan bahwa model keputusan non-autoregresif menghadirkan solusi yang kokoh, deterministik, dan sangat efisien. Beroperasi pada latensi sub-200ms dengan nol token emisi, model seperti TypeSafe JEV System One, OpenJev 0,5B, dan Kev-0,8B meraih akurasi perutean hingga 100%, kepatuhan skema 100%, dan tingkat kelulusan SLA 100% pada biaya \$0,05 per 100.000 kueri—memangkas 99,87% biaya operasional dibandingkan LLM 8B.

Lebih lanjut, pengujian empiris lintas arsitektur pada **Pipeline Tandem Hibrida Dua-Tingkat** melintasi Sahabat-AI 8B, Qwen 2.5 7B, dan Gemma 2 9B pada akselerator ganda NVIDIA Tesla T4 membuktikan tercapainya kondisi Pareto Optimal: Tingkat 1 (Model Keputusan) memberangkatkan armada penyelamat dalam 178,5 ms hingga 217,7 ms ($<$220 ms, menghadirkan percepatan pengiriman armada hingga 64,2$\times$ dengan akurasi 100% dan 0 token), sementara Tingkat 2 menyintesis bimbingan verbal pertolongan pertama (207–294 token) secara asinkron di luar jalur kritis tanpa menunda armada. Model keputusan juga menyaring panggilan iseng dalam 223,5 ms, membuang 95,1% beban komputasi yang sia-sia. Transformasi digital pelayanan publik nasional harus beralih ke arsitektur dua-tingkat terdekopling: mempercayakan aksi operasional seketika pada model keputusan refleks non-autoregresif, sembari mencadangkan LLM generatif fondasional untuk penalaran dan bimbingan empatik asinkron.

---

## Daftar Pustaka
1. J. Achiam *et al.*, "GPT-4 technical report," *arXiv preprint arXiv:2303.08774*, 2023.
2. A. Touvron *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
3. GoTo and Indosat Ooredoo Hutchison, "Sahabat-AI: Indonesian sovereign large language models," *Technical Whitepaper*, 2024.
4. Qwen Team, "Qwen2.5 technical report," *Alibaba Cloud Technical Report*, 2024.
5. A. Vaswani *et al.*, "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.
6. R. Y. Aminabadi *et al.*, "DeepSpeed-inference: Enabling efficient inference of Transformer models at unprecedented scale," in *IEEE/ACM Int. Conf. High Perform. Comput., Netw., Storage Anal. (SC)*, 2022, pp. 1–15.
7. B. T. Willard and R. Louf, "Efficient guided generation for large language models," *arXiv preprint arXiv:2307.09702*, 2023.
8. J. Gu, J. Bradbury, C. Xiong, V. O. Li, and R. Socher, "Non-autoregressive neural machine translation," in *Int. Conf. Learn. Represent. (ICLR)*, 2018.
9. B. Warner *et al.*, "ModernBERT: Bringing BERT into the modern era of deep learning," *Answer.AI & LightOn Technical Report*, 2024.
10. P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with disentangled attention," in *Int. Conf. Learn. Represent. (ICLR)*, 2021.
11. TypeSafe AI Research, "System One: Sub-200ms non-autoregressive decision architectures for enterprise triage," *TypeSafe Whitepaper*, 2025.
12. L. V. Green, "Queueing analysis in healthcare," in *Patient Flow: Reducing Delay in Healthcare Delivery*. Boston, MA: Springer, 2006, pp. 281–307.
13. A. Mandelbaum, W. A. Massey, and M. I. Reiman, "Strong approximations for Markovian service networks," *Queueing Syst.*, vol. 30, no. 1, pp. 149–201, 1998.
14. P. A. Zandbergen, "Accuracy of municipal dispatch and emergency response under high-density queuing," *IEEE Trans. Eng. Manage.*, vol. 69, no. 4, pp. 1120–1134, 2022.
15. M. S. Eisenberg and T. J. Mengert, "Cardiac resuscitation," *New England Journal of Medicine*, vol. 344, no. 17, pp. 1304–1313, 2001.
16. Gemma Team, Google DeepMind, "Gemma 2: Improving open language models at a practical size," *arXiv preprint arXiv:2408.00118*, 2024.
