import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Vaka: Zihin Sarayı Mimarı", layout="wide")

st.title("🧠 Vaka: Zihin Sarayı Mimarı (Yapay Nöron)")
st.markdown("""
**Sherlock'un Notu:** "Beyin sadece elektrikle çalışan bir karar makinesidir. 
Moriarty'yi yakalamak için zihnimdeki bağlantıları (sinapsları) doğru ağırlıklarla bağlamalıyım. 
Eğer bağlantı zayıfsa sinyal geçmez, güçlüyse karar tetiklenir."

**Görev:** Bu tek bir nöronu eğiteceğiz. Hedefimiz: Sadece **iki ipucu da VARSA** (1, 1) alarm çalsın. Diğer durumlarda sussun. (Mantıksal 'VE' Kapısı).
""")

# --- YAN PANEL: SİNAPS AYARLARI ---
with st.sidebar:
    st.header("🛠️ Sinaps (Ağırlık) Ayarları")
    
    mode = st.radio("Mod Seç:", ["Manuel Ayar (Sen Yap)", "Otomatik Öğrenme (Yapay Zeka)"])
    
    if mode == "Manuel Ayar (Sen Yap)":
        w1 = st.slider("Ağırlık 1 (Ayakkabı Çamurlu mu?)", -5.0, 5.0, 0.5)
        w2 = st.slider("Ağırlık 2 (Gergin mi?)", -5.0, 5.0, 0.5)
        bias = st.slider("Eşik Değeri (Bias - Önyargı)", -5.0, 5.0, -1.0)
        learning_rate = 0 # Manuel modda kullanılmaz
    else:
        st.info("Kör Dağcı (Gradient Descent) algoritması bu ayarları senin yerine yapacak.")
        lr = st.slider("Öğrenme Hızı", 0.01, 1.0, 0.1)
        epochs = st.slider("Eğitim Turu", 10, 500, 100)
        
        if st.button("Beyni Eğit 🧠"):
            st.session_state['train'] = True
        else:
            st.session_state['train'] = False
            
        # Başlangıç değerleri (Rastgele)
        w1, w2, bias = 0.5, 0.5, -0.5 # Default görsel için

# --- MATEMATİK MOTORU (NÖRON) ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neuron_decision(i1, i2, w1, w2, b):
    # Nöronun Formülü: Z = (x1*w1) + (x2*w2) + b
    z = (i1 * w1) + (i2 * w2) + b
    # Aktivasyon (0 ile 1 arasına sıkıştır)
    return sigmoid(z)

# --- EĞİTİM MOTORU (KÖR DAĞCI ALGORİTMASI) ---
def train_neuron(epochs, lr):
    # Veri Seti (AND Kapısı)
    # Girdi 1, Girdi 2 -> Beklenen Sonuç
    data = [
        (0, 0, 0), # Temiz, Sakin -> SUÇSUZ (0)
        (0, 1, 0), # Temiz, Gergin -> SUÇSUZ (0)
        (1, 0, 0), # Çamurlu, Sakin -> SUÇSUZ (0)
        (1, 1, 1), # Çamurlu, Gergin -> SUÇLU (1) ! HEDEF BU
    ]
    
    # Rastgele Başlangıç Ağırlıkları
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()
    
    history = []
    
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        total_error = 0
        for i1, i2, target in data:
            # 1. İleri Yayılım (Tahmin Et)
            pred = neuron_decision(i1, i2, w1, w2, b)
            
            # 2. Hata Ne? (Kör Dağcı'nın Yüksekliği)
            error = pred - target
            total_error += error**2
            
            # 3. Geri Yayılım (Ağırlıkları Güncelle - Türev)
            # Zincir kuralı basitleştirilmiş hali:
            w1 -= lr * error * i1
            w2 -= lr * error * i2
            b -= lr * error
            
        history.append(total_error)
        if epoch % 10 == 0:
            progress_bar.progress(epoch / epochs)
            
    return w1, w2, b, history

# --- ANA AKIŞ ---

if mode == "Otomatik Öğrenme (Yapay Zeka)" and st.session_state.get('train'):
    w1, w2, bias, loss_history = train_neuron(epochs, lr)
    st.success(f"Eğitim Tamamlandı! Nöron Öğrendi. \n Yeni Ağırlıklar: w1={w1:.2f}, w2={w2:.2f}, bias={bias:.2f}")
    
    # Hata Grafiği
    st.line_chart(loss_history)
    st.caption("Zamanla azalan hata oranı (Kör dağcı vadiye iniyor!)")

# --- GÖRSELLEŞTİRME (ZİHİN SARAYI) ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔬 Nöronun İç Yapısı")
    
    # Çizim Alanı
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    
    # Nöronları Çiz
    circle_in1 = plt.Circle((0, 2), 0.3, color='skyblue', zorder=10)
    circle_in2 = plt.Circle((0, 0), 0.3, color='skyblue', zorder=10)
    circle_out = plt.Circle((3, 1), 0.5, color='orange', zorder=10)
    
    ax.add_patch(circle_in1)
    ax.add_patch(circle_in2)
    ax.add_patch(circle_out)
    
    # Etiketler
    ax.text(-0.8, 2, "Girdi 1\n(Çamur)", fontsize=12)
    ax.text(-0.8, 0, "Girdi 2\n(Gerginlik)", fontsize=12)
    ax.text(3.6, 1, "ÇIKTI\n(Karar)", fontsize=12)
    
    # Bağlantılar (Weights) - Kalınlık ağırlığa göre değişir
    # w1 çizgisi
    width1 = abs(w1) * 2
    color1 = 'green' if w1 > 0 else 'red'
    ax.plot([0, 3], [2, 1], color=color1, linewidth=width1, alpha=0.6)
    ax.text(1.5, 1.8, f"w1: {w1:.2f}", fontsize=10, color=color1, fontweight='bold')
    
    # w2 çizgisi
    width2 = abs(w2) * 2
    color2 = 'green' if w2 > 0 else 'red'
    ax.plot([0, 3], [0, 1], color=color2, linewidth=width2, alpha=0.6)
    ax.text(1.5, 0.2, f"w2: {w2:.2f}", fontsize=10, color=color2, fontweight='bold')
    
    # Bias (Nöronun içinde)
    ax.text(2.6, 1, f"Bias\n{bias:.2f}", fontsize=9, ha='center', color='white')

    st.pyplot(fig)

with col2:
    st.subheader("🕵️‍♂️ Sherlock Testi")
    st.write("Bakalım Nöron doğru karar veriyor mu?")
    
    # Test Senaryoları
    scenarios = [
        (0, 0, "Temiz & Sakin"),
        (1, 0, "Çamurlu & Sakin"),
        (0, 1, "Temiz & Gergin"),
        (1, 1, "Çamurlu & Gergin (SUÇLU!)")
    ]
    
    for i1, i2, label in scenarios:
        result = neuron_decision(i1, i2, w1, w2, bias)
        
        # Karar Görseli
        decision_text = "HAPİS 🚨" if result > 0.8 else "SERBEST 🟢"
        bar_color = "red" if result > 0.8 else "green"
        
        st.write(f"**{label}**")
        st.progress(float(result))
        st.caption(f"Şüphe Oranı: %{result*100:.1f} -> Karar: {decision_text}")
        st.divider()

    with st.expander("👨‍🏫 Mennan Usta'nın Yorumu"):
        st.write("""
        "Bak evlat, Manuel Mod'da ayarları tutturmak zor, değil mi? 
        
        İşte 'Yapay Zeka' dediğimiz şey, o sürgüleri (w1, w2) bizim yerimize milyonlarca kez deneyip en doğrusunu bulan sabırlı bir çıraktan başka bir şey değil."
        """)
