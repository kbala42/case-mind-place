import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.title("🧠 Vaka 4: Siber Nöron")
    
    # Kilit kontrolünü test için pasif bırakabiliriz veya açabiliriz
    # if 'inventory_coordinates' not in st.session_state: st.error("Kilitli"); return

    if 'math_mode_4' not in st.session_state: st.session_state['math_mode_4'] = False
    st.markdown("**Görev:** Sadece (Çamur + Gerginlik) varsa ateş et. (AND Kapısı)" if not st.session_state['math_mode_4'] else "### 📐 Perceptron & Sigmoid")

    with st.sidebar:
        mode = st.radio("Mod:", ["Manuel", "Otomatik (AI)"])
        if mode == "Manuel":
            w1 = st.slider("w1", -5.0, 5.0, 0.5); w2 = st.slider("w2", -5.0, 5.0, 0.5); b = st.slider("bias", -5.0, 5.0, -1.0)
        else:
            if st.button("Eğit"): st.session_state['train_neuron'] = True
            w1, w2, b = 1.0, 1.0, -1.5 # Default

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    
    if mode == "Otomatik (AI)" and st.session_state.get('train_neuron'):
        # Basit eğitim simülasyonu
        w1, w2, b = 5.0, 5.0, -8.0 # İdeal AND kapısı değerlerine yakın
        st.success("AI Eğitildi!")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3)); ax.axis('off')
        ax.text(0, 0.8, "Girdi 1"); ax.text(0, 0.2, "Girdi 2"); ax.text(1, 0.5, "KARAR")
        ax.plot([0.2, 0.8], [0.8, 0.5], lw=abs(w1), c='g' if w1>0 else 'r')
        ax.plot([0.2, 0.8], [0.2, 0.5], lw=abs(w2), c='g' if w2>0 else 'r')
        st.pyplot(fig)

    with col2:
        fp, fn = 0, 0
        data = [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]
        for i1, i2, t in data:
            pred = 1 if sigmoid(i1*w1 + i2*w2 + b) > 0.5 else 0
            if t==0 and pred==1: fp+=1
            if t==1 and pred==0: fn+=1
        
        st.metric("False Positive (Masum Yandı)", fp)
        st.metric("False Negative (Suçlu Kaçtı)", fn)
        if fp > 0: st.error("Masumları yaktın!")

    st.divider()
    if st.button("🔴 Kırmızı Hap"):
        st.session_state['math_mode_4'] = not st.session_state['math_mode_4']
        if hasattr(st, "rerun"): st.rerun() 
        else: st.experimental_rerun()

if __name__ == "__main__":
    run()