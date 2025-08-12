# -*- coding: utf-8 -*-
import os, glob, joblib
import pandas as pd
import streamlit as st

# ================== AYARLAR ==================
FINAL_MODEL_DIR   = "final_choice_models"     # FinalChoice .pkl klasörü
FEATURE_CSV       = "feature_defaults.csv"    # (opsiyonel) profil dosyası
REQUIRED_FEATURES = ["energy","integral_error","Central_Value","Absolute_PDF_Uncertainty"]
ALL_PROCESSES     = ["LO","NLO","NNLO"]

st.set_page_config(page_title="Cross-Section Tahmin (Süreç Bazlı Giriş)", layout="centered")
st.title("🔬 Cross-Section Prediction")
st.markdown("<small>Created by **Merve Balki**</small>", unsafe_allow_html=True)

# ================== MODELLER ==================
@st.cache_resource(show_spinner=False)
def load_models(model_dir=FINAL_MODEL_DIR):
    """
    Dosya adı: <process>_<channel>_<ModelName>_Final.pkl
    Örn: LO_91_Lasso_Final.pkl
    """
    models = {}
    channels = set()
    pattern = os.path.join(model_dir, "*_Final.pkl")
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        parts = fname.split("_")
        if len(parts) < 4:
            continue
        proc, ch = parts[0], parts[1]
        model_name = "_".join(parts[2:-1])
        try:
            mdl = joblib.load(path)
        except Exception as e:
            st.warning(f"Model yüklenemedi: {fname} ({e})")
            continue
        models[(str(proc), str(ch))] = (model_name, mdl)
        channels.add(str(ch))
    return models, sorted(channels, key=lambda x: (len(x), x))

# ================== PROFİL (opsiyonel) ==================
@st.cache_data(show_spinner=False)
def load_feature_profiles(csv_path=FEATURE_CSV):
    if not os.path.exists(csv_path):
        return None
    prof = pd.read_csv(csv_path)
    need = {"process","channel","energy","integral_error","Central_Value","Absolute_PDF_Uncertainty"}
    miss = need - set(prof.columns)
    if miss:
        st.warning(f"`{csv_path}` eksik kolon(lar): {sorted(miss)} — profil kullanılmayacak.")
        return None
    prof["process"] = prof["process"].astype(str)
    prof["channel"] = prof["channel"].astype(str)
    return prof.sort_values(["process","channel","energy"]).reset_index(drop=True)

def interpolate_features(prof_df: pd.DataFrame, process: str, channel: str, energy: float):
    sub = prof_df[(prof_df["process"]==str(process)) & (prof_df["channel"]==str(channel))]
    if sub.empty:
        return None
    exact = sub[sub["energy"]==energy]
    if len(exact):
        r = exact.iloc[0]
        return {
            "energy": float(energy),
            "integral_error": float(r["integral_error"]),
            "Central_Value": float(r["Central_Value"]),
            "Absolute_PDF_Uncertainty": float(r["Absolute_PDF_Uncertainty"]),
        }
    lower = sub[sub["energy"] < energy].tail(1)
    upper = sub[sub["energy"] > energy].head(1)
    if len(lower) and len(upper):
        l, u = lower.iloc[0], upper.iloc[0]
        t = (energy - l["energy"]) / (u["energy"] - l["energy"])
        lerp = lambda a,b: (1-t)*a + t*b
        return {
            "energy": float(energy),
            "integral_error": float(lerp(l["integral_error"], u["integral_error"])),
            "Central_Value": float(lerp(l["Central_Value"], u["Central_Value"])),
            "Absolute_PDF_Uncertainty": float(lerp(l["Absolute_PDF_Uncertainty"], u["Absolute_PDF_Uncertainty"])),
        }
    r = (lower.iloc[0] if len(lower) else upper.iloc[0])
    return {
        "energy": float(energy),
        "integral_error": float(r["integral_error"]),
        "Central_Value": float(r["Central_Value"]),
        "Absolute_PDF_Uncertainty": float(r["Absolute_PDF_Uncertainty"]),
    }

# ================== YÜKLE ==================
models_map, available_channels = load_models()
if not models_map:
    st.error(f"`{FINAL_MODEL_DIR}` klasöründe FinalChoice model dosyası bulunamadı.")
    st.stop()
profiles = load_feature_profiles()  # olmayabilir (None)

# ================== ARAYÜZ ==================
st.subheader("⚙️ INPUT")
channel = st.selectbox("Kanal", options=available_channels, index=0)
energy  = st.number_input("Energy (tüm süreçler için ortak)", min_value=0.0, value=100.0, step=1.0)

# Süreç seçimi: kullanıcı bir veya birden çok süreç seçebilir
selected_processes = st.multiselect("Süreç(ler) seçin", options=ALL_PROCESSES, default=ALL_PROCESSES)

# Süreç başına giriş tarzı (manuel mi otomatik mi?) ve manuel ise değerler
per_proc_cfg = {}
for proc in selected_processes:
    with st.expander(f"🔧 {proc} için giriş seçenekleri", expanded=False):
        use_manual = st.checkbox(f"{proc}: Manuel feature girişi", value=False, key=f"man_{proc}")
        if use_manual:
            ie  = st.number_input(f"{proc} • integral_error", value=0.0, step=0.001, format="%.6f", key=f"ie_{proc}")
            cv  = st.number_input(f"{proc} • Central_Value", value=float(energy), step=1.0, key=f"cv_{proc}")
            apu = st.number_input(f"{proc} • Absolute_PDF_Uncertainty", value=0.0, step=0.001, format="%.6f", key=f"apu_{proc}")
            per_proc_cfg[proc] = {"mode":"manual", "integral_error":ie, "Central_Value":cv, "Absolute_PDF_Uncertainty":apu}
        else:
            per_proc_cfg[proc] = {"mode":"auto"}  # profil/enterpolasyon (varsa); yoksa fallback

st.markdown("---")
if st.button("🔮 Tahmin Et"):
    if not selected_processes:
        st.warning("En az bir süreç seçmelisin.")
        st.stop()

    rows = []
    missing_any = []

    for proc in selected_processes:
        key = (proc, str(channel))
        if key not in models_map:
            missing_any.append(proc)
            continue

        cfg = per_proc_cfg.get(proc, {"mode":"auto"})
        if cfg["mode"] == "manual":
            feats = {
                "energy": float(energy),
                "integral_error": float(cfg["integral_error"]),
                "Central_Value": float(cfg["Central_Value"]),
                "Absolute_PDF_Uncertainty": float(cfg["Absolute_PDF_Uncertainty"]),
            }
        else:
            feats = None
            if profiles is not None:
                feats = interpolate_features(profiles, proc, channel, float(energy))
            if feats is None:
                # profil yoksa makul fallback: CV=energy, diğerleri 0
                feats = {
                    "energy": float(energy),
                    "integral_error": 0.0,
                    "Central_Value": float(energy),
                    "Absolute_PDF_Uncertainty": 0.0,
                }

        X = pd.DataFrame([feats], columns=REQUIRED_FEATURES)
        mdl_name, mdl = models_map[key]
        try:
            yhat = float(mdl.predict(X)[0])
        except Exception as e:
            st.warning(f"{proc}-{channel} ({mdl_name}) tahmin hatası: {e}")
            continue

        rows.append({
            "Process": proc,
            "Channel": str(channel),
            "Model": mdl_name,
            "Energy": float(energy),
            "integral_error": feats["integral_error"],
            "Central_Value": feats["Central_Value"],
            "Absolute_PDF_Uncertainty": feats["Absolute_PDF_Uncertainty"],
            "Predicted Cross Section": yhat,
        })

    if missing_any:
        st.info("FinalChoice modeli bulunamadı (atlandı): " + ", ".join(missing_any))

    if not rows:
        st.error("Seçimlerin için tahmin üretilemedi. Model/profil dosyalarını kontrol et.")
        st.stop()

    out = pd.DataFrame(rows).set_index("Process").reindex(selected_processes).reset_index()

    st.subheader(f"🏁 Kanal: {channel} — Energy: {energy}")
    st.table(out[["Process","Channel","Model","Predicted Cross Section"]])

    with st.expander("Kullanılan feature değerleri (süreç bazında)"):
        st.dataframe(out[["Process","Energy","integral_error","Central_Value","Absolute_PDF_Uncertainty"]],
                     use_container_width=True)

else:
    st.info("Kanalı ve energy’yi seç → süreç(ler)i işaretle → her süreç için giriş tarzını belirle → **Tahmin Et**.")