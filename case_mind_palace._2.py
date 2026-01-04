import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def run():
    st.title("🧠 Vaka 4: Siber Nöron (Zihin Sarayı)")

    # --- 1. BAĞLANTI KONTROLÜ (Story Arc) ---
    # Vaka 3'ten (Kör Dağcı) öğrenme yeteneğini kazanmış olması lazım.
    # Basitlik için Vaka 2'den gelen koordinat var mı diye bakıyoruz.
    if 'inventory_coordinates' not in st.session_state:
        st.error("⛔ ERİŞİM ENGELLENDİ: Dedektif, henüz Vadi'ye inmedin (Vaka 3). Önce optimizasyon eğitimini tamamla.")
        return

    st.success("✅ Erişim İzni Verildi: Optimizasyon Modülü Aktif.")

    # --- 2. HİKAYE / MATEMATİK MODU ---
    if 'math_mode_4' not in st.session_state:
        st.session_state['math_mode_4'] = False

    if not st.session_state['math_mode_4']:
        st.markdown("""
        **Görev:** Moriarty'nin ajanlarını tespit eden bir "Karar Çipi" yapıyoruz.
        Kural net: **Sadece ve Sadece 2 Tehdit (Çamur + Gerginlik) AYNI ANDA varsa ateş et.** (AND Kapısı).
        Eğer tek bir şüphede ateş edersen, masumları vurursun!
        """)
    else:
        st.markdown("""
        ### 📐 MATEMATİKSEL YÜZLEŞME
        **Konu:** Yapay Nöron (Perceptron) & Aktivasyon Fonksiyonu
        
        Mennan Usta'nın "Ağırlıklı Karar" dediği şey, matematikte **Dot Product + Sigmoid** işlemidir:
        
        $$ z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + b $$
        $$ \hat{y} = \frac{1}{1 + e^{-z}} $$
        
        * $w$: Ağırlık (Önem derecesi).
        * $b$: Bias (Önyargı/Eşik).
        * Sigmoid: Sonucu 0 ile 1 arasına sıkıştırır (Olasılık).
        """)

    # --- YAN PANEL: SİNAPS AYARLARI ---
    with st.sidebar:
        st.header("🛠️ Sinaps Ayarları")
        mode = st.radio("Eğitim Modu:", ["Manuel Ayar (Sen Yap)", "Otomatik Öğrenme (AI)"])
        
        if mode == "Manuel Ayar (Sen Yap)":
            w1 = st.slider("w1 (Çamur)", -5.0, 5.0, 0.5)
            w2 = st.slider("w2 (Gerginlik)", -5.0, 5.0, 0.5)
            bias = st.slider("Bias (Eşik)", -5.0, 5.0, -1.0)
            lr = 0
            epochs = 0
        else:
            st.info("Kör Dağcı algoritması (Vaka 3) burada devreye girecek.")
            lr = st.slider("Öğrenme Hızı", 0.01, 1.0, 0.1)
            epochs = st.slider("Eğitim Turu", 10, 500, 100)
            
            if st.button("Beyni Eğit 🧠"):
                st.session_state['train_neuron'] = True
            
            # Başlangıç değerleri (Rastgelelik hissi için)
            w1, w2, bias = 1.0, 1.0, -1.5 

    # --- FONKSİYONLAR ---
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def neuron_decision(i1, i2, w1, w2, b):
        z = (i1 * w1) + (i2 * w2) + b
        return sigmoid(z)

    def train_neuron(epochs, lr):
        data = [
            (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1) # AND Kapısı
        ]
        # Rastgele Başlangıç
        w1 = np.random.randn()
        w2 = np.random.randn()
        b = np.random.randn()
        
        history = []
        progress_bar = st.progress(0)
        
        for epoch in range(epochs):
            total_error = 0
            for i1, i2, target in data:
                pred = neuron_decision(i1, i2, w1, w2, b)
                error = pred - target
                total_error += error**2
                
                # Gradient Descent (Türev)
                w1 -= lr * error * i1
                w2 -= lr * error * i2
                b -= lr * error
            
            history.append(total_error)
            if epoch % 10 == 0:
                progress_bar.progress(epoch / epochs)
        
        return w1, w2, b, history

    # --- ANA AKIŞ ---
    if mode == "Otomatik Öğrenme (AI)" and st.session_state.get('train_neuron'):
        w1, w2, bias, loss = train_neuron(epochs, lr)
        st.success(f"Eğitim Bitti! w1={w1:.2f}, w2={w2:.2f}, bias={bias:.2f}")
        st.line_chart(loss)

    # --- GÖRSELLEŞTİRME VE TEST ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔬 Nöronun İç Yapısı")
        
        # Basit Matplotlib Çizimi
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(-1, 4); ax.set_ylim(-1, 3); ax.axis('off')
        
        # Nöronlar
        circle_in1 = plt.Circle((0, 2), 0.3, color='skyblue')
        circle_in2 = plt.Circle((0, 0), 0.3, color='skyblue')
        circle_out = plt.Circle((3, 1), 0.5, color='orange')
        ax.add_patch(circle_in1); ax.add_patch(circle_in2); ax.add_patch(circle_out)
        
        # Etiketler
        ax.text(-0.8, 2, "Girdi 1", fontsize=10)
        ax.text(-0.8, 0, "Girdi 2", fontsize=10)
        ax.text(3.6, 1, "Karar", fontsize=10)
        
        # Bağlantılar (Renk değişimi ağırlığa göre)
        c1 = 'green' if w1 > 0 else 'red'
        c2 = 'green' if w2 > 0 else 'red'
        ax.plot([0, 3], [2, 1], color=c1, linewidth=abs(w1)*2, alpha=0.6)
        ax.plot([0, 3], [0, 1], color=c2, linewidth=abs(w2)*2, alpha=0.6)
        
        ax.text(1.5, 1.8, f"w1: {w1:.2f}", color=c1, fontweight='bold')
        ax.text(1.5, 0.2, f"w2: {w2:.2f}", color=c2, fontweight='bold')
        ax.text(2.6, 1, f"b:{bias:.1f}", color='white', ha='center', fontsize=8)
        
        st.pyplot(fig)

    with col2:
        st.subheader("🕵️‍♂️ Sherlock Testi")
        
        # Confusion Matrix Hazırlığı için sayaçlar
        tp, tn, fp, fn = 0, 0, 0, 0
        
        scenarios = [
            (0, 0, 0, "Temiz & Sakin"),
            (0, 1, 0, "Temiz & Gergin"),
            (1, 0, 0, "Çamurlu & Sakin"),
            (1, 1, 1, "Çamurlu & Gergin")
        ]
        
        for i1, i2, target, label in scenarios:
            prob = neuron_decision(i1, i2, w1, w2, bias)
            pred = 1 if prob > 0.5 else 0
            
            # Matris Hesapla
            if target == 1 and pred == 1: tp += 1
            elif target == 0 and pred == 0: tn += 1
            elif target == 0 and pred == 1: fp += 1 # MASUMU YAKTIK!
            elif target == 1 and pred == 0: fn += 1 # SUÇLU KAÇTI!
            
            st.write(f"**{label}**")
            st.progress(float(prob))
            
    # --- 3. YENİ EKLENTİ: VİCDAN MATRİSİ (Confusion Matrix) ---
    st.divider()
    st.subheader("⚖️ Vicdan Muhasebesi (Confusion Matrix)")
    
    cm_col1, cm_col2 = st.columns([1, 2])
    
    with cm_col1:
        st.write("Yapay Zeka ne kadar adil davrandı?")
        st.write(f"🟢 **Doğru Karar:** {tp + tn}")
        st.write(f"🔴 **Hatalı Karar:** {fp + fn}")
        
        if fp > 0:
            st.error(f"😱 DİKKAT: {fp} Masum kişiyi suçlu sandın! (False Positive)")
        if fn > 0:
            st.warning(f"⚠️ DİKKAT: {fn} Suçlu elinden kaçtı! (False Negative)")
            
    with cm_col2:
        # Basit Isı Haritası
        matrix = np.array([[tn, fp], [fn, tp]])
        fig_cm, ax_cm = plt.subplots(figsize=(4, 2))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Masum', 'Suçlu'], yticklabels=['Masum', 'Suçlu'])
        ax_cm.set_xlabel("Tahmin")
        ax_cm.set_ylabel("Gerçek")
        st.pyplot(fig_cm)

    # --- 4. REALITY CHECK & MATH TOGGLE ---
    st.divider()
    if st.button("🔴 Kırmızı Hap: Analojiyi Kır"):
        st.session_state['math_mode_4'] = not st.session_state['math_mode_4']
        st.rerun()

    with st.expander("🛠️ Kod Müdahalesi (Reality Check)"):
        st.write("**Soru:** Eğer `Bias` değerini çok yüksek bir pozitif sayı (+5.0) yaparsan ne olur?")
        ans = st.radio("Cevap:", ["Hiçbir şey değişmez", "Nöron sürekli 'SUÇLU' der (Aşırı Duyarlı)", "Nöron hiç çalışmaz"])
        
        if ans == "Nöron sürekli 'SUÇLU' der (Aşırı Duyarlı)":
            st.success("Doğru! Bias eşiği çok düşürür (veya pozitif destek verir), en ufak sinyalde bile ateşleme yapar.")
        elif ans:
            st.error("Yanlış. Bias pozitifse, nöronun ateşlenmesi kolaylaşır.")

if __name__ == "__main__":
    run()