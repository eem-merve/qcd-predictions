# -*- coding: utf-8 -*-
import os, io, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import LeaveOneOut, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ------------------- UI -------------------
st.set_page_config(page_title="Cross-Section ML Pipeline", layout="wide")
st.title("🔬 Cross-Section ML Web Pipeline (Streamlit)")

with st.sidebar:
    st.header("⚙️ Parametreler")
    TEST_ENERGY = st.number_input("Test energy (hold-out)", min_value=0, value=100, step=1)
    OUTLIER_THRESH = st.number_input("SafeEnsemble MAPE eşiği (%)", min_value=0.0, value=3.0, step=0.5)
    MAE_FALLBACK = st.number_input("true≈0 fallback: MAE <% of median(|y|)", min_value=0.0, value=5.0, step=0.5) / 100.0
    SEED = st.number_input("Seed", min_value=0, value=42, step=1)

    do_save_all = st.checkbox("Tüm modelleri .pkl kaydet", value=True)
    do_save_final = st.checkbox("FinalChoice modellerini de kaydet", value=True)
    run_loo = st.checkbox("LOO CV (tüm eğitim) hesapla", value=False)
    plot_learning_curves = st.checkbox("Learning Curves çiz", value=False)

    st.markdown("---")
    uploaded = st.file_uploader("📥 Excel veri yükleyin (`.xlsx`)", type=["xlsx"])
    default_path = st.text_input("Veya dosya yolu", value="data_all.xlsx")

    # Kayıt konumu: çalışma klasörü (kalıcı)
    workdir = os.getcwd()
    save_all_dir = os.path.join(workdir, "saved_models")
    save_final_dir = os.path.join(workdir, "final_choice_models")
    os.makedirs(save_all_dir, exist_ok=True)
    os.makedirs(save_final_dir, exist_ok=True)

run_btn = st.sidebar.button("🚀 Pipeline'ı Çalıştır")

# ------------------- Helpers -------------------
def get_models(seed: int):
    return {
        'Linear': Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        'Ridge':  Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))]),
        'Lasso':  Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=1e-3, random_state=seed, max_iter=10000))]),
        'BayesianRidge': Pipeline([("scaler", StandardScaler()), ("bayes", BayesianRidge())]),
        'KNN':    Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=3, weights="distance"))]),
        'Poly2':  Pipeline([("poly", PolynomialFeatures(2, include_bias=False)), ("scaler", StandardScaler(with_mean=False)), ("reg", LinearRegression())]),
        'Poly3':  Pipeline([("poly", PolynomialFeatures(3, include_bias=False)), ("scaler", StandardScaler(with_mean=False)), ("reg", LinearRegression())]),
        'XGB':    XGBRegressor(objective='reg:squarederror', random_state=seed,
                               max_depth=3, learning_rate=0.1, n_estimators=300,
                               reg_alpha=0.5, reg_lambda=0.5, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=-1),
        'RF':     RandomForestRegressor(n_estimators=400, max_depth=6, random_state=seed, n_jobs=-1)
    }

def loo_cv_score(model, X, y):
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr_idx, te_idx in loo.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model.fit(X_tr, y_tr)
        preds.append(float(model.predict(X_te)[0]))
        trues.append(float(y_te.values[0]))
    preds, trues = np.array(preds), np.array(trues)
    denom = np.where(np.abs(trues) < 1e-12, np.nan, np.abs(trues))
    mape = np.nanmean(np.abs((preds - trues) / denom)) * 100.0
    return {'MAE': mean_absolute_error(trues, preds),
            'MAPE': mape,
            'RMSE': np.sqrt(mean_squared_error(trues, preds)),
            'R2': r2_score(trues, preds)}

def plot_learning_curve_fig(model, X, y, title="Learning Curve", cv=5, seed=42):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_absolute_error',
        train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=seed, n_jobs=-1
    )
    tr_mean = -train_scores.mean(axis=1); va_mean = -val_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(train_sizes, tr_mean, 'o-', label='Train MAE')
    ax.plot(train_sizes, va_mean, 's-', label='Validation MAE')
    ax.set_xlabel('Eğitim seti boyutu'); ax.set_ylabel('MAE'); ax.set_title(title)
    ax.legend(); ax.grid(True, ls=':'); fig.tight_layout()
    return fig

def safe_ensemble(row, model_keys, mape_thresh, mae_fallback, y_median_abs):
    safe_models, safe_preds = [], []
    y_true = row['true_value']
    for key in model_keys:
        pred_val = row[f'{key}_pred']
        mape_val = row.get(f'{key}_abs_percent_error', np.inf)
        mae_val  = abs(row.get(f'{key}_abs_error', np.inf))

        good = False
        if np.isfinite(y_true) and abs(y_true) > 1e-12:
            if np.isfinite(mape_val) and (mape_val < mape_thresh):
                good = True
        else:
            if np.isfinite(mae_val) and (mae_val <= mae_fallback * y_median_abs):
                good = True

        if good and np.isfinite(pred_val) and (pred_val >= 0):
            safe_models.append(key); safe_preds.append(pred_val)

    if len(safe_preds) == 0:
        return row['XGB_pred'], 1, 'XGB'
    return float(np.mean(safe_preds)), len(safe_preds), ', '.join(safe_models)

def _row_metric(row, key, y_median_abs):
    pe = row.get(f'{key}_abs_percent_error', np.nan)
    if np.isfinite(pe): return float(pe)
    ae = abs(row.get(f'{key}_abs_error', np.inf))
    denom = y_median_abs if y_median_abs > 0 else 1.0
    return 100.0 * (ae / denom)

def pick_best_model_vs_ensemble(row, model_keys, y_median_abs):
    ens_metric = _row_metric(row, 'SafeEnsemble', y_median_abs)
    best_key, best_metric = None, np.inf
    for key in model_keys:
        m = _row_metric(row, key, y_median_abs)
        if np.isfinite(m) and m < best_metric:
            best_metric, best_key = m, key
    if np.isfinite(best_metric) and best_metric < ens_metric:
        chosen_key, pred = best_key, row[f'{best_key}_pred']
    else:
        chosen_key, pred = 'SafeEnsemble', row['SafeEnsemble_pred']
    abs_error = float(pred - row['true_value'])
    denom = abs(row['true_value']) if abs(row['true_value']) > 1e-12 else np.nan
    abs_percent_error = 100.0 * abs(abs_error) / denom if np.isfinite(denom) else np.nan
    return pred, chosen_key, abs_error, abs_percent_error

def df_to_excel_download_bytes(df: pd.DataFrame, sheet_name="Sheet1"):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    out.seek(0); return out

# ------------------- Run -------------------
if run_btn:
    try:
        # Veri yükleme
        if uploaded is not None:
            data = pd.read_excel(uploaded)
            st.success(f"Veri yüklendi: **{uploaded.name}** (satır: {len(data)})")
        else:
            if not os.path.exists(default_path):
                st.error("Dosya bulunamadı. Lütfen bir `.xlsx` yükleyin veya geçerli bir yol girin.")
                st.stop()
            data = pd.read_excel(default_path)
            st.success(f"Veri yüklendi: **{default_path}** (satır: {len(data)})")

        required_cols = {'energy','cross_section','process','channel',
                         'integral_error','Central_Value','Absolute_PDF_Uncertainty'}
        missing = required_cols - set(map(str, data.columns))
        if missing:
            st.error(f"Eksik zorunlu sütun(lar): {sorted(missing)}")
            st.stop()

        input_features = ['energy','integral_error','Central_Value','Absolute_PDF_Uncertainty']
        train_data = data[data['energy'] != TEST_ENERGY]
        test_data  = data[data['energy'] == TEST_ENERGY]
        if len(test_data)==0:
            st.warning(f"Test verisi yok (energy == {TEST_ENERGY}). Dosyayı kontrol edin.")
            st.stop()

        # >>> DÜZELTME: process_levels ve channels birleştirmesi set/union ile
        process_levels = sorted(
            set(train_data['process'].dropna().astype(str)) |
            set(test_data['process'].dropna().astype(str))
        )
        channels = sorted(
            set(train_data['channel'].dropna().astype(str)) |
            set(test_data['channel'].dropna().astype(str))
        )

        model_keys = list(get_models(SEED).keys())
        all_rows, manifest_records = [], []

        with st.status("Eğitim • Tahmin • Kayıt işlemleri çalışıyor…", expanded=True) as status:
            for process in process_levels:
                for channel in channels:
                    tr_ch = train_data[(train_data['process'].astype(str)==process)&(train_data['channel'].astype(str)==channel)]
                    te_ch = test_data[(test_data['process'].astype(str)==process)&(test_data['channel'].astype(str)==channel)]
                    tr_ch = tr_ch.dropna(subset=input_features+['cross_section'])
                    te_ch = te_ch.dropna(subset=input_features+['cross_section'])
                    if len(tr_ch)==0 or len(te_ch)==0:
                        st.write(f"⏭️ Atlandı: {process}-{channel} (yetersiz veri)")
                        continue

                    X_tr, y_tr = tr_ch[input_features].copy(), tr_ch['cross_section'].copy()
                    X_te, y_te = te_ch[input_features].copy(), te_ch['cross_section'].copy()

                    preds = {}
                    for key, mdl in get_models(SEED).items():
                        mdl.fit(X_tr, y_tr)
                        preds[key] = mdl.predict(X_te)

                        if do_save_all:
                            fname = f"{process}_{channel}_{key}.pkl"
                            fpath = os.path.join(save_all_dir, fname)
                            joblib.dump(mdl, fpath)
                            manifest_records.append({
                                "process": process, "channel": channel, "model": key,
                                "path": fpath, "n_train": len(tr_ch)
                            })

                    for i, y_t in enumerate(y_te):
                        row = {"process": process, "channel": channel, "energy": TEST_ENERGY,
                               "true_value": float(y_t), "n_train": len(tr_ch)}
                        for key in model_keys:
                            row[f'{key}_pred'] = float(preds[key][i])
                        all_rows.append(row)

            status.update(label="Eğitim ve tahminler tamam ✅", state="complete")

        # Ensemble + Final
        final_df = pd.DataFrame(all_rows)
        if len(final_df) == 0:
            st.error("Hiç satır oluşmadı. Veri ve TEST_ENERGY seçimini kontrol et.")
            st.stop()

        y_abs = np.abs(final_df['true_value'])
        y_abs_safe = y_abs.replace(0, np.nan)
        for key in model_keys:
            final_df[f'{key}_abs_error'] = final_df[f'{key}_pred'] - final_df['true_value']
            final_df[f'{key}_abs_percent_error'] = 100.0 * np.abs(final_df[f'{key}_abs_error']) / y_abs_safe

        ensemble_preds, ensemble_n, ensemble_models = [], [], []
        y_median_abs = float(np.nanmedian(y_abs)) if len(y_abs) else 1.0
        for _, row in final_df.iterrows():
            ens_pred, n_used, used = safe_ensemble(
                row, model_keys, OUTLIER_THRESH, MAE_FALLBACK, y_median_abs if y_median_abs>0 else 1.0
            )
            ensemble_preds.append(ens_pred); ensemble_n.append(n_used); ensemble_models.append(used)

        final_df['SafeEnsemble_pred'] = ensemble_preds
        final_df['SafeEnsemble_n_models'] = ensemble_n
        final_df['SafeEnsemble_models'] = ensemble_models
        final_df['SafeEnsemble_abs_error'] = final_df['SafeEnsemble_pred'] - final_df['true_value']
        final_df['SafeEnsemble_abs_percent_error'] = 100.0 * np.abs(final_df['SafeEnsemble_abs_error']) / y_abs_safe

        final_pred, final_src, final_ae, final_ape = [], [], [], []
        for _, row in final_df.iterrows():
            p, s, ae, ape = pick_best_model_vs_ensemble(row, model_keys, y_median_abs)
            final_pred.append(p); final_src.append(s); final_ae.append(ae); final_ape.append(ape)

        final_df['Final_pred'] = final_pred
        final_df['Final_source'] = final_src
        final_df['Final_abs_error'] = final_ae
        final_df['Final_abs_percent_error'] = final_ape

        # FinalChoice modellerini kaydet (process-channel bazında)
        final_manifest = []
        saved_final_models = set()
        if do_save_final:
            for (proc, ch), grp in final_df.groupby(['process','channel']):
                src = grp['Final_source'].iloc[0]
                if src == 'SafeEnsemble':
                    continue
                key_tuple = (proc, ch, src)
                if key_tuple in saved_final_models:
                    continue
                tr_ch = train_data[(train_data['process'].astype(str)==proc) & (train_data['channel'].astype(str)==ch)]
                tr_ch = tr_ch.dropna(subset=input_features + ['cross_section'])
                if len(tr_ch)==0:
                    continue
                X_tr = tr_ch[input_features].copy()
                y_tr = tr_ch['cross_section'].copy()
                mdl = get_models(SEED)[src]
                mdl.fit(X_tr, y_tr)
                fname = f"{proc}_{ch}_{src}_Final.pkl"
                fpath = os.path.join(save_final_dir, fname)
                joblib.dump(mdl, fpath)
                final_manifest.append({"process": proc, "channel": ch, "final_model": src, "path": fpath, "n_train": len(tr_ch)})
                saved_final_models.add(key_tuple)

        # Özet tablolar
        summary = []
        for key in model_keys + ['SafeEnsemble','Final']:
            if key == 'Final':
                ae, ape = final_df['Final_abs_error'], final_df['Final_abs_percent_error']
            else:
                ae, ape = final_df[f'{key}_abs_error'], final_df[f'{key}_abs_percent_error']
            summary.append({'Model': key,
                            'MAE': float(np.mean(np.abs(ae))),
                            'MAPE': float(np.nanmean(ape)),
                            'RMSE': float(np.sqrt(np.mean(ae**2)))})
        summary_df = pd.DataFrame(summary).sort_values(by='MAE')

        st.subheader("📊 Model Performans Özeti")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("📄 Satır Bazında Sonuçlar (ilk 200)")
        st.dataframe(final_df.head(200), use_container_width=True)

        # İndirme butonları
        st.markdown("### ⬇️ Çıktıları İndir")
        results_xlsx = df_to_excel_download_bytes(final_df, sheet_name="results")
        summary_xlsx = df_to_excel_download_bytes(summary_df, sheet_name="summary")
        st.download_button("Sonuç tablosu (XLSX)", results_xlsx, file_name=f"regressor_safeensemble_finalchoice_testenergy{TEST_ENERGY}_nolog.xlsx")
        st.download_button("Özet tablosu (XLSX)", summary_xlsx, file_name=f"model_summary_finalchoice_testenergy{TEST_ENERGY}_nolog.xlsx")

        # Manifest’ler
        if do_save_all and len(manifest_records):
            manifest_df = pd.DataFrame(manifest_records)
            st.download_button("Tüm modeller manifest (CSV)", manifest_df.to_csv(index=False).encode("utf-8"),
                               file_name="saved_models_manifest.csv")
            st.success(f"📁 Tüm modeller kaydedildi → {save_all_dir}")
        if do_save_final and len(final_manifest):
            final_manifest_df = pd.DataFrame(final_manifest)
            st.download_button("FinalChoice manifest (CSV)", final_manifest_df.to_csv(index=False).encode("utf-8"),
                               file_name="final_choice_models_manifest.csv")
            st.success(f"🏁 FinalChoice modelleri kaydedildi → {save_final_dir}")

        # Grafik
        st.subheader("🖼️ True vs Predicted (Final)")
        fig_scatter, ax = plt.subplots(figsize=(8,6))
        ax.scatter(final_df['true_value'], final_df['Final_pred'], label='Final', alpha=0.85, marker='X')
        mn, mx = float(final_df['true_value'].min()), float(final_df['true_value'].max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=2, label='y = x')
        ax.set_xlabel('Gerçek Değer'); ax.set_ylabel('Tahmin Edilen Değer')
        ax.grid(True, ls=':'); ax.legend(); fig_scatter.tight_layout()
        st.pyplot(fig_scatter)

        # Learning Curves (opsiyonel)
        if plot_learning_curves:
            st.subheader("📈 Learning Curves")
            X_all = data[data['energy'] != TEST_ENERGY].dropna(subset=input_features + ['cross_section'])[input_features]
            y_all = data[data['energy'] != TEST_ENERGY].dropna(subset=input_features + ['cross_section'])['cross_section']
            st.pyplot(plot_learning_curve_fig(get_models(SEED)['XGB'], X_all, y_all, title="Learning Curve (XGBoost)", seed=SEED))
            st.pyplot(plot_learning_curve_fig(get_models(SEED)['Linear'], X_all, y_all, title="Learning Curve (Linear)", seed=SEED))

        # LOO (opsiyonel)
        if run_loo:
            st.subheader("🔁 LOO CV (tüm eğitim)")
            X_all = data[data['energy'] != TEST_ENERGY].dropna(subset=input_features + ['cross_section'])[input_features]
            y_all = data[data['energy'] != TEST_ENERGY].dropna(subset=input_features + ['cross_section'])['cross_section']
            loo_rows = []
            for key in get_models(SEED).keys():
                res = loo_cv_score(get_models(SEED)[key], X_all, y_all)
                loo_rows.append({"Model": key, **res})
            loo_df = pd.DataFrame(loo_rows).sort_values(by="MAE")
            st.dataframe(loo_df, use_container_width=True)

        st.success("Bitti ✅ Yeni dosyayla/parametreyle tekrar çalıştırabilirsin.")

    except Exception as e:
        st.error(f"Hata: {e}")
        st.exception(e)
else:
    st.write("👈 Soldan dosyayı yükleyip **Pipeline'ı Çalıştır** düğmesine basın.")
    st.caption("Gerekli sütunlar: energy, cross_section, process, channel, integral_error, Central_Value, Absolute_PDF_Uncertainty")