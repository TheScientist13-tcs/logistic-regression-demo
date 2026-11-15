import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
)
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = "DejaVu Sans"

st.set_page_config(
    page_title="Logistic Regression ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    .main .block-container {
        padding-top: 0.9rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }
    h1 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem !important;
        padding-bottom: 0.1rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.04rem !important;
        font-weight: 700 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.69rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.82;
        white-space: nowrap !important;
    }
    div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }
    div[data-testid="stMetricDelta"] {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.72rem !important;
    }
    .section-label {
        background: rgba(148, 163, 184, 0.10);
        border-left: 3px solid #64748B;
        color: inherit;
        border-radius: 3px;
        padding: 4px 10px;
        margin-top: 4px;
        margin-bottom: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .status-banner {
        background: rgba(148, 163, 184, 0.08);
        border-left: 3px solid #475569;
        color: inherit;
        border-radius: 3px;
        padding: 6px 12px;
        margin-top: 4px;
        margin-bottom: 6px;
        font-size: 0.77rem;
        line-height: 1.4;
    }
    .interp-card {
        background: rgba(15, 23, 42, 0.40);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-left: 3px solid #64748B;
        border-radius: 4px;
        padding: 8px 12px;
        margin-top: 4px;
        margin-bottom: 6px;
        font-size: 0.75rem;
        line-height: 1.45;
        color: inherit;
    }
    .interp-card ul {
        margin: 4px 0 0 0;
        padding-left: 16px;
    }
    .interp-card li {
        margin-bottom: 2px;
    }
    .katex-display {
        margin: 0.18em 0 !important;
        overflow-x: auto;
        overflow-y: hidden;
    }
    .katex-display .katex {
        font-size: 0.85rem !important;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.73rem;
        margin-top: 2px;
        margin-bottom: 6px;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }
    .data-table th {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        padding: 4px 6px;
        text-align: center;
        font-weight: 600;
        white-space: nowrap;
        border-bottom: 1px solid rgba(148, 163, 184, 0.30);
    }
    .data-table td {
        padding: 3px 6px;
        text-align: center;
        white-space: nowrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.72rem;
        color: #F1F5F9 !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    }
    .data-table tr.row-odd td {
        background-color: #0F172A !important;
    }
    .data-table tr.row-even td {
        background-color: #1E293B !important;
    }
    .data-table tr.row-mis td {
        background-color: #3B1824 !important;
        color: #FECDD3 !important;
        font-weight: 600;
    }
    .cm-tp-tn {
        background-color: rgba(16, 185, 129, 0.18) !important;
        color: #6EE7B7 !important;
        font-weight: 600;
    }
    .cm-fp-fn {
        background-color: rgba(244, 63, 94, 0.18) !important;
        color: #FDA4AF !important;
        font-weight: 600;
    }
    .text-ok {
        color: #34D399 !important;
        font-weight: 600;
    }
    .text-err {
        color: #FB7185 !important;
        font-weight: 600;
    }
    .text-neutral {
        color: #FBBF24 !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"] {
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SYNTHETIC_DATASETS = {
    "1. Shifted & Tilted Corridor (Separable)": {
        "X_train": np.array(
            [
                [2.4, 2.8],
                [1.9, 2.6],
                [1.3, 2.4],
                [0.6, 2.1],
                [0.0, 1.9],
                [-0.7, 1.5],
                [-1.5, 1.1],
                [1.9, 1.7],
                [1.2, 1.3],
                [0.5, 1.0],
                [0.0, 0.8],
                [-0.7, 0.4],
                [-1.8, -0.9],
                [-2.5, -1.8],
            ]
        ),
        "y_train": np.array([1.0] * 7 + [0.0] * 7),
        "X_test": np.array(
            [
                [1.5, 2.5],
                [-0.2, 1.8],
                [-1.1, 1.3],
                [1.5, 1.4],
                [-0.3, 0.6],
                [-1.4, -0.6],
            ]
        ),
        "y_test": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "mode": "separable",
    },
    "2. Overlapping Clusters (Boundary-Zone Overlap)": {
        "X_train": np.array(
            [
                [2.0, 2.70],
                [1.0, 2.35],
                [0.0, 2.00],
                [-1.0, 1.65],
                [-2.0, 1.30],
                [0.4, 1.25],
                [-0.4, 0.35],
                [-0.4, 0.97],
                [0.4, 0.63],
                [2.0, 0.30],
                [1.0, -0.05],
                [0.0, -0.40],
                [-1.0, -0.75],
                [-2.0, -1.10],
            ]
        ),
        "y_train": np.array([1.0] * 7 + [0.0] * 7),
        "X_test": np.array(
            [
                [1.5, 2.55],
                [-1.2, 1.60],
                [-0.8, 0.82],
                [0.2, 1.18],
                [1.3, 0.15],
                [-1.5, -0.95],
            ]
        ),
        "y_test": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "mode": "overlapping",
    },
    "3. Margin Outliers & Boundary Skew (Overfitting)": {
        "X_train": np.array(
            [
                [2.3, 2.2],
                [1.2, 1.9],
                [0.1, 1.6],
                [-0.8, 1.6],
                [-1.9, 1.9],
                [1.1, -0.3],
                [2.0, -0.6],
                [-2.0, 0.6],
                [-1.1, 0.3],
                [1.9, -1.9],
                [0.8, -1.6],
                [-0.1, -1.6],
                [-1.2, -1.9],
                [-2.3, -2.2],
            ]
        ),
        "y_train": np.array([1.0] * 7 + [0.0] * 7),
        "X_test": np.array(
            [
                [1.5, 1.5],
                [-0.4, 1.2],
                [-1.6, 0.4],
                [1.6, -0.4],
                [0.4, -1.2],
                [-1.5, -1.5],
            ]
        ),
        "y_test": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "mode": "overfitting",
    },
    "4. Doughnut / Concentric Rings (Underfitting)": {
        "X_train": np.array(
            [
                [0.5, 0.7],
                [-0.4, 0.8],
                [-0.5, 0.2],
                [0.4, 0.2],
                [0.0, 0.9],
                [0.0, 0.5],
                [0.0, 0.1],
                [2.2, 1.2],
                [0.9, 2.3],
                [-0.9, 2.2],
                [-2.1, 1.2],
                [-2.2, -1.1],
                [-0.2, -1.5],
                [1.8, -0.5],
            ]
        ),
        "y_train": np.array([1.0] * 7 + [0.0] * 7),
        "X_test": np.array(
            [
                [0.2, 0.6],
                [-0.2, 0.5],
                [0.1, 0.3],
                [1.6, 1.9],
                [-1.7, -0.1],
                [0.9, -1.4],
            ]
        ),
        "y_test": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "mode": "doughnut",
    },
    "5. Pure Intercept Translation (Parallel Shift)": {
        "X_train": np.array(
            [
                [1.6, 1.6],
                [2.2, 0.8],
                [0.8, 2.2],
                [1.2, 1.2],
                [1.8, 0.4],
                [0.4, 1.8],
                [1.5, 0.9],
                [0.4, 0.4],
                [0.9, -0.1],
                [-0.1, 0.9],
                [0.1, 0.5],
                [-0.6, -0.6],
                [-1.4, -0.4],
                [-0.4, -1.4],
            ]
        ),
        "y_train": np.array([1.0] * 7 + [0.0] * 7),
        "X_test": np.array(
            [
                [1.4, 1.0],
                [1.0, 1.4],
                [0.8, 0.8],
                [0.2, 0.2],
                [-0.4, 0.2],
                [0.2, -0.4],
            ]
        ),
        "y_test": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "mode": "separable",
    },
}


def fmt_delta(val: float, dec: int = 1, unit: str = "") -> str:
    if np.isnan(val):
        return "Undef"
    if val > 1e-12:
        return f"+{val:.{dec}f}{unit}"
    if val < -1e-12:
        return f"{val:.{dec}f}{unit}"
    return f"{0.0:.{dec}f}{unit}"


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(a, -50, 50)))


def logit_threshold(tau: float) -> float:
    tau_clipped = float(np.clip(tau, 1e-6, 1.0 - 1e-6))
    return float(np.log(tau_clipped / (1.0 - tau_clipped)))


def compute_mean_nll_from_probs(probs: np.ndarray, y_vec: np.ndarray) -> float:
    p_clip = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_vec * np.log(p_clip) + (1.0 - y_vec) * np.log(1.0 - p_clip)))


def boundary_geometry(w_vec: np.ndarray, tau: float = 0.5):
    logit_tau = logit_threshold(tau)
    if abs(w_vec[2]) > 1e-6:
        slope = float(-w_vec[1] / w_vec[2])
        intercept = float((logit_tau - w_vec[0]) / w_vec[2])
    else:
        slope = float("nan")
        intercept = float("nan")
    if np.hypot(w_vec[1], w_vec[2]) > 1e-6:
        angle = float(np.degrees(np.arctan2(w_vec[2], w_vec[1])))
    else:
        angle = float("nan")
    return slope, intercept, angle


def compute_roc_curve_and_auc(probs: np.ndarray, y_true: np.ndarray, tau: float = 0.5):
    pos_scores = probs[y_true == 1.0]
    neg_scores = probs[y_true == 0.0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    diff = pos_scores[:, None] - neg_scores[None, :]
    auc_val = float(
        (np.sum(diff > 1e-12) + 0.5 * np.sum(np.abs(diff) <= 1e-12)) / (n_pos * n_neg)
    )

    uniq_thresh = np.sort(np.unique(np.append(probs, float(tau))))[::-1]
    tpr_vals = np.sum(pos_scores[:, None] >= uniq_thresh[None, :], axis=0) / n_pos
    fpr_vals = np.sum(neg_scores[:, None] >= uniq_thresh[None, :], axis=0) / n_neg

    fpr_arr = np.concatenate(([0.0], fpr_vals, [1.0]))
    tpr_arr = np.concatenate(([0.0], tpr_vals, [1.0]))
    return fpr_arr, tpr_arr, auc_val


def compute_pr_curve_and_ap(probs: np.ndarray, y_true: np.ndarray, tau: float = 0.5):
    y_pos_mask = y_true == 1.0
    y_neg_mask = ~y_pos_mask
    n_pos = int(np.sum(y_pos_mask))
    total = len(y_true)
    pos_prior = float(n_pos / total) if total > 0 else 0.5

    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0, pos_prior

    score_thresh = np.sort(np.unique(probs))[::-1]
    pred_pos_ap = probs[:, None] >= score_thresh[None, :]
    tp_ap = np.sum(pred_pos_ap & y_pos_mask[:, None], axis=0, dtype=float)
    fp_ap = np.sum(pred_pos_ap & y_neg_mask[:, None], axis=0, dtype=float)
    denom_ap = tp_ap + fp_ap
    ap_rec_arr = np.concatenate(([0.0], tp_ap / n_pos))
    ap_prec_arr = np.concatenate(
        ([1.0], np.divide(tp_ap, denom_ap, out=np.ones_like(tp_ap), where=denom_ap > 0))
    )
    ap_val = float(np.sum((ap_rec_arr[1:] - ap_rec_arr[:-1]) * ap_prec_arr[1:]))

    plot_thresh = np.sort(np.unique(np.append(probs, float(tau))))[::-1]
    pred_pos_pl = probs[:, None] >= plot_thresh[None, :]
    tp_pl = np.sum(pred_pos_pl & y_pos_mask[:, None], axis=0, dtype=float)
    fp_pl = np.sum(pred_pos_pl & y_neg_mask[:, None], axis=0, dtype=float)
    denom_pl = tp_pl + fp_pl
    rec_arr = np.concatenate(([0.0], tp_pl / n_pos))
    prec_arr = np.concatenate(
        ([1.0], np.divide(tp_pl, denom_pl, out=np.ones_like(tp_pl), where=denom_pl > 0))
    )

    return rec_arr, prec_arr, ap_val, pos_prior


@st.cache_data(max_entries=512, show_spinner=False)
def compute_classification_metrics(
    probs: np.ndarray, y_true: np.ndarray, nll_val: float, tau: float = 0.5
) -> dict:
    preds_bin = probs >= tau
    y_pos = y_true == 1.0
    y_neg = ~y_pos

    tp = int(np.sum(preds_bin & y_pos))
    fp = int(np.sum(preds_bin & y_neg))
    fn = int(np.sum((~preds_bin) & y_pos))
    tn = int(np.sum((~preds_bin) & y_neg))

    total = tp + fp + fn + tn
    acc = (tp + tn) / total * 100.0 if total > 0 else 0.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    fpr_arr, tpr_arr, auc = compute_roc_curve_and_auc(probs, y_true, tau=tau)
    rec_arr, prec_arr, ap, pos_prior = compute_pr_curve_and_ap(probs, y_true, tau=tau)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "spec": spec,
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "pos_prior": pos_prior,
        "nll": nll_val,
        "fpr_curve": fpr_arr,
        "tpr_curve": tpr_arr,
        "rec_curve": rec_arr,
        "prec_curve": prec_arr,
        "fpr_op": (fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tpr_op": (tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "prec_op": (tp / (tp + fp)) if (tp + fp) > 0 else 1.0,
    }


def sample_random_w0() -> np.ndarray:
    rng = np.random.default_rng()
    w1 = float(rng.uniform(-0.9, 0.7))
    w2 = float(rng.uniform(0.55, 1.25))
    b0 = float(rng.uniform(-1.2, 1.4))
    w_bias = -b0 * w2
    return np.round(np.array([w_bias, w1, w2], dtype=float), 2)


@st.cache_data(max_entries=128, show_spinner=False)
def simulate_core_trajectory(
    X_tr_tilde: np.ndarray,
    y_tr: np.ndarray,
    X_te_tilde: np.ndarray,
    y_te: np.ndarray,
    w_init_tuple: tuple,
    eta: float,
    max_iter: int,
    normalize_grad: bool = False,
):
    steps = []
    w_curr = np.array(w_init_tuple, dtype=float)
    n_tr = float(len(y_tr))
    scale = n_tr if normalize_grad else 1.0

    logits_tr = X_tr_tilde @ w_curr
    preds_tr = sigmoid(logits_tr)
    logits_te = X_te_tilde @ w_curr
    preds_te = sigmoid(logits_te)
    tr_nll_t = compute_mean_nll_from_probs(preds_tr, y_tr)
    te_nll_t = compute_mean_nll_from_probs(preds_te, y_te)

    for t in range(max_iter + 1):
        errors_tr = preds_tr - y_tr
        grad = (X_tr_tilde.T @ errors_tr) / scale
        errors_te = preds_te - y_te

        w_next = w_curr - eta * grad
        logits_tr_next = X_tr_tilde @ w_next
        preds_tr_next = sigmoid(logits_tr_next)
        logits_te_next = X_te_tilde @ w_next
        preds_te_next = sigmoid(logits_te_next)

        tr_nll_next = compute_mean_nll_from_probs(preds_tr_next, y_tr)
        te_nll_next = compute_mean_nll_from_probs(preds_te_next, y_te)

        steps.append(
            {
                "t": t,
                "w_t": w_curr,
                "w_next": w_next,
                "grad": grad,
                "logits_tr": logits_tr,
                "preds_tr": preds_tr,
                "errors_tr": errors_tr,
                "preds_tr_next": preds_tr_next,
                "logits_te": logits_te,
                "preds_te": preds_te,
                "errors_te": errors_te,
                "preds_te_next": preds_te_next,
                "tr_nll_t": tr_nll_t,
                "tr_nll_next": tr_nll_next,
                "te_nll_t": te_nll_t,
                "te_nll_next": te_nll_next,
            }
        )

        w_curr = w_next
        logits_tr = logits_tr_next
        preds_tr = preds_tr_next
        logits_te = logits_te_next
        preds_te = preds_te_next
        tr_nll_t = tr_nll_next
        te_nll_t = te_nll_next

    return steps


@st.cache_data(max_entries=512, show_spinner=False)
def get_step_state(
    X_tr_tilde: np.ndarray,
    y_tr: np.ndarray,
    X_te_tilde: np.ndarray,
    y_te: np.ndarray,
    w_init_tuple: tuple,
    eta: float,
    t: int,
    tau: float = 0.5,
    normalize_grad: bool = False,
):
    horizon = ((t // 50) + 1) * 50
    core_steps = simulate_core_trajectory(
        X_tr_tilde,
        y_tr,
        X_te_tilde,
        y_te,
        w_init_tuple,
        eta,
        horizon,
        normalize_grad=normalize_grad,
    )
    raw = core_steps[t]

    metrics_tr_t = compute_classification_metrics(
        raw["preds_tr"], y_tr, raw["tr_nll_t"], tau=tau
    )
    metrics_tr_next = compute_classification_metrics(
        raw["preds_tr_next"], y_tr, raw["tr_nll_next"], tau=tau
    )
    metrics_te_t = compute_classification_metrics(
        raw["preds_te"], y_te, raw["te_nll_t"], tau=tau
    )
    metrics_te_next = compute_classification_metrics(
        raw["preds_te_next"], y_te, raw["te_nll_next"], tau=tau
    )

    m_t, b_t, ang_t = boundary_geometry(raw["w_t"], tau=tau)
    m_next, b_next, ang_next = boundary_geometry(raw["w_next"], tau=tau)
    _, b_next_05, _ = boundary_geometry(raw["w_next"], tau=0.5)

    return {
        **raw,
        "metrics_tr_t": metrics_tr_t,
        "metrics_tr_next": metrics_tr_next,
        "metrics_te_t": metrics_te_t,
        "metrics_te_next": metrics_te_next,
        "m_t": m_t,
        "b_t": b_t,
        "ang_t": ang_t,
        "m_next": m_next,
        "b_next": b_next,
        "b_next_05": b_next_05,
        "ang_next": ang_next,
    }


@st.cache_data(show_spinner=False)
def generate_sklearn_2d_split(
    gen_name: str,
    n_samples: int,
    noise_val: float,
    test_ratio: float,
    random_seed: int,
):
    if "make_moons" in gen_name:
        X_raw, y_raw = make_moons(
            n_samples=n_samples, noise=noise_val, random_state=random_seed
        )
        X_centered = X_raw - np.mean(X_raw, axis=0)
        X_centered[:, 1] *= -1.0
        mode_tag = "moons"
        desc = (
            "Two interleaving crescent manifolds in 2D. A linear boundary captures "
            "the coarse vertical separation (~85% accuracy) but cannot follow the non-linear curve."
        )
    elif "make_circles" in gen_name:
        X_raw, y_raw = make_circles(
            n_samples=n_samples,
            noise=noise_val,
            factor=0.42,
            random_state=random_seed,
        )
        X_centered = X_raw * 1.65
        mode_tag = "doughnut"
        desc = (
            "Concentric 2D rings where the inner core (y = 1) is surrounded by the outer ring (y = 0). "
            "Illustrates high-bias underfitting (ROC-AUC near 0.50)."
        )
    elif "make_blobs" in gen_name:
        X_raw, y_raw = make_blobs(
            n_samples=n_samples,
            centers=[[-1.05, -0.95], [1.05, 0.95]],
            cluster_std=0.45 + 1.6 * noise_val,
            random_state=random_seed,
        )
        X_centered = X_raw
        mode_tag = "blobs"
        desc = (
            "Two isotropic 2D Gaussian clusters with controllable standard deviation. "
            "Demonstrates smooth transition from separable clusters to boundary-zone overlap."
        )
    else:
        X_raw, y_raw = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            flip_y=min(noise_val * 0.4, 0.20),
            class_sep=max(1.45 - 1.8 * noise_val, 0.45),
            random_state=random_seed,
        )
        X_centered = X_raw
        if np.mean(X_centered[y_raw == 1, 1]) < np.mean(X_centered[y_raw == 0, 1]):
            X_centered[:, 1] *= -1.0
        mode_tag = "classification"
        desc = (
            "Anisotropic 2D Gaussian clusters generated via make_classification with "
            "correlated features and label noise."
        )

    y_float = y_raw.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_centered,
        y_float,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=y_float,
    )

    return {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_test": X_te,
        "y_test": y_te,
        "X_train_tilde": np.hstack([np.ones((len(y_tr), 1)), X_tr]),
        "X_test_tilde": np.hstack([np.ones((len(y_te), 1)), X_te]),
        "mode": mode_tag,
        "description": desc,
    }


def build_sample_table_html(
    X_coords,
    y_labels,
    logits,
    preds,
    errors,
    w_curr,
    t_step: int,
    tau: float = 0.5,
    prefix: str = "tr",
    start_idx: int = 1,
):
    rows = []
    is_neutral = np.allclose(w_curr, 0.0) and np.isclose(tau, 0.5)
    for i in range(len(X_coords)):
        idx_num = start_idx + i
        base_parity = "row-even" if i % 2 == 1 else "row-odd"
        if is_neutral:
            status_html = '<span class="text-neutral">0.50</span>'
            row_cls = base_parity
        else:
            pred_label = 1.0 if preds[i] >= tau else 0.0
            is_correct = pred_label == y_labels[i]
            status_html = (
                '<span class="text-ok">Correct</span>'
                if is_correct
                else '<span class="text-err">Error</span>'
            )
            row_cls = base_parity if is_correct else "row-mis"

        rows.append(
            f'<tr class="{row_cls}">'
            f"<td><b>{prefix}<sub>{idx_num}</sub></b></td>"
            f"<td>({X_coords[i,0]:.2f}, {X_coords[i,1]:.2f})</td>"
            f"<td>{int(y_labels[i])}</td>"
            f"<td>{logits[i]:.2f}</td>"
            f"<td>{preds[i]:.2f}</td>"
            f"<td>{errors[i]:.2f}</td>"
            f"<td>{status_html}</td>"
            f"</tr>"
        )
    rows_html = "".join(rows)
    return (
        f'<table class="data-table">'
        f"<thead>"
        f"<tr>"
        f"<th>Sample</th>"
        f"<th>(x₁, x₂)</th>"
        f"<th>y</th>"
        f"<th>aₙ</th>"
        f"<th>ŷₙ</th>"
        f"<th>ŷₙ − y</th>"
        f"<th>At w<sup>({t_step})</sup> (τ={tau:.2f})</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )


def build_metrics_comparison_html(
    m_tr_t: dict,
    m_tr_next: dict,
    m_te_t: dict,
    m_te_next: dict,
    t_step: int,
    tau: float,
    n_tr: int,
    n_te: int,
) -> str:
    tr_pct = n_tr / (n_tr + n_te) * 100.0
    te_pct = n_te / (n_tr + n_te) * 100.0

    rows_spec = [
        (f"Accuracy @ τ={tau:.2f}", "acc", "%", 1, False),
        (f"Precision (PPV) @ τ={tau:.2f}", "prec", "%", 1, False),
        (f"Recall (TPR) @ τ={tau:.2f}", "rec", "%", 1, False),
        (f"Specificity (TNR) @ τ={tau:.2f}", "spec", "%", 1, False),
        (f"F1-Score @ τ={tau:.2f}", "f1", "%", 1, False),
        ("ROC-AUC [τ-Invariant]", "auc", "", 3, False),
        ("PR-AUC / AP [τ-Invariant]", "ap", "", 3, False),
        ("Mean NLL Loss [τ-Invariant]", "nll", "", 4, True),
    ]

    rows = []
    for idx, (label, key, unit, dec, lower_is_better) in enumerate(rows_spec):
        row_cls = "row-even" if idx % 2 == 1 else "row-odd"
        tr_val = m_tr_next[key]
        tr_d = m_tr_next[key] - m_tr_t[key]
        te_val = m_te_next[key]
        te_d = m_te_next[key] - m_te_t[key]

        gap = tr_val - te_val

        if not lower_is_better:
            is_test_worse = gap > 1e-9
        else:
            is_test_worse = te_val > (tr_val + 1e-9)

        gap_cls = "text-err" if is_test_worse else "text-ok"
        tr_d_str = fmt_delta(tr_d, dec, unit)
        te_d_str = fmt_delta(te_d, dec, unit)

        rows.append(
            f'<tr class="{row_cls}">'
            f"<td style='text-align:left; padding-left:8px;'><b>{label}</b></td>"
            f"<td><b>{tr_val:.{dec}f}{unit}</b> <span style='opacity:0.75;'>({tr_d_str})</span></td>"
            f"<td><b>{te_val:.{dec}f}{unit}</b> <span style='opacity:0.75;'>({te_d_str})</span></td>"
            f'<td><span class="{gap_cls}">{gap:.{dec}f}{unit}</span></td>'
            f"</tr>"
        )

    rows_html = "".join(rows)
    return (
        f'<table class="data-table">'
        f"<thead>"
        f"<tr>"
        f'<th style="text-align:left; padding-left:8px;">Metric at w<sup>({t_step+1})</sup> (Δ from w<sup>({t_step})</sup>)</th>'
        f"<th>Train (N={n_tr} | {tr_pct:.0f}%)</th>"
        f"<th>Test (N={n_te} | {te_pct:.0f}%)</th>"
        f"<th>Gap (Train − Test)</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )


def build_confusion_matrix_html(m_dict: dict, title_str: str, tau: float = 0.5) -> str:
    return (
        f'<table class="data-table">'
        f"<thead>"
        f'<tr><th colspan="3">{title_str}</th></tr>'
        f"<tr>"
        f"<th>Actual \\ Predicted</th>"
        f"<th>ŷ ≥ {tau:.2f} (Pos)</th>"
        f"<th>ŷ < {tau:.2f} (Neg)</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>"
        f'<tr class="row-odd">'
        f"<th>y = 1 (Pos)</th>"
        f'<td class="cm-tp-tn">TP = {m_dict["tp"]}</td>'
        f'<td class="cm-fp-fn">FN = {m_dict["fn"]}</td>'
        f"</tr>"
        f'<tr class="row-even">'
        f"<th>y = 0 (Neg)</th>"
        f'<td class="cm-fp-fn">FP = {m_dict["fp"]}</td>'
        f'<td class="cm-tp-tn">TN = {m_dict["tn"]}</td>'
        f"</tr>"
        f"</tbody>"
        f"</table>"
    )


def build_boundary_interpretation_html(
    t_step: int,
    m_next: float,
    b_next: float,
    b_next_05: float,
    tau: float,
    logit_tau_val: float,
    m_tr_next: dict,
    m_te_next: dict,
    n_tr: int,
    n_te: int,
    mode: str,
) -> str:
    tr_err = m_tr_next["fp"] + m_tr_next["fn"]
    te_err = m_te_next["fp"] + m_te_next["fn"]
    slope_str = f"{m_next:.3f}" if not np.isnan(m_next) else "Undefined"
    int_str = f"{b_next:.3f}" if not np.isnan(b_next) else "Undefined"
    ref_str = f"{b_next_05:.3f}" if not np.isnan(b_next_05) else "Undefined"

    extra_note = ""
    if mode == "doughnut":
        extra_note = (
            "<li><b>Non-Linear Ring Topology:</b> Because the positive inner core (<i>y</i> = 1) is surrounded by the negative outer ring (<i>y</i> = 0), "
            "a single hyperplane cannot isolate the core—sliding <b>τ</b> sweeps the half-plane across both rings simultaneously.</li>"
        )
    elif mode == "moons":
        extra_note = (
            "<li><b>Interleaving Crescent Manifolds:</b> Logistic regression finds the best linear compromise across the two moons, "
            "leaving the curved tips of each crescent misclassified unless non-linear basis functions are introduced.</li>"
        )

    return (
        f'<div class="interp-card">'
        f"<b>Decision Boundary Interpretation at w<sup>({t_step+1})</sup> (τ = {tau:.2f} ⟺ log-odds aₙ ≥ {logit_tau_val:.3f})</b>"
        f"<ul>"
        f"<li><b>Parallel Threshold Shift:</b> Adjusting <b>τ</b> keeps the slope fixed at <b>m = {slope_str}</b> while translating the intercept from <b>b(0.50) = {ref_str}</b> to <b>b({tau:.2f}) = {int_str}</b>.</li>"
        f"<li><b>Active Error Breakdown:</b> At <b>τ = {tau:.2f}</b>, the linear boundary misclassifies <b>{tr_err}/{n_tr}</b> training samples (<b>FP = {m_tr_next['fp']}, FN = {m_tr_next['fn']}</b>) and <b>{te_err}/{n_te}</b> test samples (<b>FP = {m_te_next['fp']}, FN = {m_te_next['fn']}</b>).</li>"
        f"{extra_note}"
        f"</ul>"
        f"</div>"
    )


def build_roc_interpretation_html(
    t_step: int, m_tr_next: dict, m_te_next: dict, tau: float, mode: str
) -> str:
    tr_pos = m_tr_next["tp"] + m_tr_next["fn"]
    tr_neg = m_tr_next["fp"] + m_tr_next["tn"]
    te_pos = m_te_next["tp"] + m_te_next["fn"]
    te_neg = m_te_next["fp"] + m_te_next["tn"]

    extra_note = ""
    if mode == "doughnut":
        extra_note = (
            "<li><b>Concentric Ring Projection (AUC ≈ 0.50–0.57):</b> Along any 1D linear projection <i>aₙ = x̃ₙᵀw<sup>(t+1)</sup></i>, "
            "both the highest and lowest scores belong to the outer negative ring (<i>y</i> = 0), while the inner positive core (<i>y</i> = 1) sits in the middle. "
            "Lowering <b>τ</b> encounters outer-ring negatives first (dipping below the diagonal), then sweeps across the inner core (rising above the diagonal).</li>"
        )

    return (
        f'<div class="interp-card">'
        f"<b>ROC Curve & AUC Interpretation (at w<sup>({t_step+1})</sup>, Active Threshold τ = {tau:.2f})</b>"
        f"<ul>"
        f"<li><b>Global Ranking Quality (τ-Invariant):</b> <b>Train ROC-AUC = {m_tr_next['auc']:.3f}</b> and <b>Test ROC-AUC = {m_te_next['auc']:.3f}</b> quantify the empirical probability <i>P</i>(ŷ<sub>+</sub> > ŷ<sub>−</sub>) across all cutoffs (0.500 = random guessing, 1.000 = complete separation).</li>"
        f"<li><b>Operating Point at τ = {tau:.2f}:</b> Train sits at <b>(FPR = {m_tr_next['fpr_op']*100:.1f}%, TPR = {m_tr_next['rec']:.1f}%)</b> [TP = {m_tr_next['tp']}/{tr_pos}, FP = {m_tr_next['fp']}/{tr_neg}]; Test sits at <b>(FPR = {m_te_next['fpr_op']*100:.1f}%, TPR = {m_te_next['rec']:.1f}%)</b> [TP = {m_te_next['tp']}/{te_pos}, FP = {m_te_next['fp']}/{te_neg}].</li>"
        f"<li><b>Threshold Sensitivity:</b> Raising <b>τ</b> moves the operating markers <b>down and left</b> (stricter positive calls, lower FPR and lower Recall); lowering <b>τ</b> moves them <b>up and right</b> (higher Recall at the cost of higher FPR).</li>"
        f"{extra_note}"
        f"</ul>"
        f"</div>"
    )


def build_pr_interpretation_html(
    t_step: int, m_tr_next: dict, m_te_next: dict, tau: float, mode: str
) -> str:
    pos_prior = m_tr_next["pos_prior"]
    extra_note = ""
    if mode == "doughnut":
        extra_note = (
            "<li><b>Underfitting on PR Curve:</b> Because outer-ring negatives (<i>y</i> = 0) occupy the most extreme positive logits in any linear direction, "
            "Precision drops toward the 0.50 baseline at high thresholds before recovering slightly as the threshold sweeps through the inner core.</li>"
        )

    return (
        f'<div class="interp-card">'
        f"<b>Precision–Recall & Average Precision Interpretation (at w<sup>({t_step+1})</sup>, Active Threshold τ = {tau:.2f})</b>"
        f"<ul>"
        f"<li><b>Global PR-AUC / AP (τ-Invariant):</b> <b>Train AP = {m_tr_next['ap']:.3f}</b> and <b>Test AP = {m_te_next['ap']:.3f}</b> summarize the step-wise area under the Precision–Recall curve relative to the prevalence baseline <b>P(y = 1) = {pos_prior:.2f}</b>.</li>"
        f"<li><b>Operating Point at τ = {tau:.2f}:</b> Train achieves <b>Precision = {m_tr_next['prec']:.1f}%, Recall = {m_tr_next['rec']:.1f}%</b> (<b>F₁ = {m_tr_next['f1']:.1f}%</b>); Test achieves <b>Precision = {m_te_next['prec']:.1f}%, Recall = {m_te_next['rec']:.1f}%</b> (<b>F₁ = {m_te_next['f1']:.1f}%</b>).</li>"
        f"<li><b>Threshold Trade-Off:</b> Raising <b>τ</b> shifts the operating point <b>leftward</b> (lower Recall, typically higher Precision); lowering <b>τ</b> shifts it <b>rightward</b> (higher Recall, lower Precision).</li>"
        f"{extra_note}"
        f"</ul>"
        f"</div>"
    )


@st.cache_data(max_entries=16, show_spinner=False)
def get_meshgrid(lo: float, hi: float, res: int = 150):
    grid_vals = np.linspace(lo, hi, res)
    xx1, xx2 = np.meshgrid(grid_vals, grid_vals)
    return xx1, xx2, np.linspace(lo, hi, 200)


@st.cache_data(max_entries=256, show_spinner=False)
def render_boundary_png(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preds_tr_next: np.ndarray,
    preds_te_next: np.ndarray,
    t_step: int,
    w_next_tuple: tuple,
    m_next: float,
    b_next: float,
    b_next_05: float,
    tau: float,
    logit_tau_val: float,
    mode: str,
    show_train_pts: bool,
    show_test_pts: bool,
    show_prob_shading: bool,
    show_pt_labels: bool,
    show_normal_arrow: bool,
    plot_bounds: tuple,
) -> bytes:
    lo, hi = plot_bounds
    w_next = np.array(w_next_tuple, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 4.15), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    xx1, xx2, x_vals = get_meshgrid(lo, hi, 150)

    if show_prob_shading:
        grid_logits = w_next[0] + w_next[1] * xx1 + w_next[2] * xx2
        grid_probs = sigmoid(grid_logits)
        ax.contourf(
            xx1,
            xx2,
            grid_probs,
            levels=np.linspace(0, 1, 25),
            cmap="RdBu",
            alpha=0.20,
            zorder=0,
        )

    ax.axhline(0, color="#CBD5E1", linewidth=1.0, zorder=1)
    ax.axvline(0, color="#CBD5E1", linewidth=1.0, zorder=1)

    if mode == "doughnut" and len(y_train) <= 20:
        theta_ring = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            0.0 + 1.15 * np.cos(theta_ring),
            0.5 + 1.15 * np.sin(theta_ring),
            color="#6366F1",
            linestyle=":",
            linewidth=1.6,
            alpha=0.75,
            label="Non-Linear Ring",
            zorder=2,
        )
    elif mode == "doughnut" and len(y_train) > 20:
        theta_ring = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            1.17 * np.cos(theta_ring),
            1.17 * np.sin(theta_ring),
            color="#6366F1",
            linestyle=":",
            linewidth=1.6,
            alpha=0.75,
            label="Ideal Radial Separator",
            zorder=2,
        )
    elif mode == "overfitting":
        ax.axhline(
            0.0,
            color="#6366F1",
            linestyle=":",
            linewidth=1.6,
            alpha=0.8,
            label="True Boundary (x₂ = 0)",
            zorder=2,
        )

    if np.hypot(w_next[1], w_next[2]) > 1e-5:
        if not np.isclose(tau, 0.5):
            if abs(w_next[2]) > 1e-4:
                y_ref_05 = -(w_next[0] + w_next[1] * x_vals) / w_next[2]
                ax.plot(
                    x_vals,
                    y_ref_05,
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.75,
                    label=f"Ref τ=0.50 (b={b_next_05:.2f})",
                    zorder=3,
                )
            else:
                ax.axvline(
                    -w_next[0] / w_next[1],
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.75,
                    label="Ref τ=0.50",
                    zorder=3,
                )

        if abs(w_next[2]) > 1e-4:
            y_curr = (logit_tau_val - w_next[0] - w_next[1] * x_vals) / w_next[2]
            ax.plot(
                x_vals,
                y_curr,
                color="#059669",
                linewidth=2.8,
                label=f"Boundary w^({t_step+1}), τ={tau:.2f} (m={m_next:.2f}, b={b_next:.2f})",
                zorder=4,
            )
        else:
            ax.axvline(
                (logit_tau_val - w_next[0]) / w_next[1],
                color="#059669",
                linewidth=2.8,
                label=f"Boundary w^({t_step+1}), τ={tau:.2f}",
                zorder=4,
            )

        if show_normal_arrow:
            norm_len = np.hypot(w_next[1], w_next[2])
            arrow_scale = (0.18 * (hi - lo)) / norm_len if norm_len > 0 else 1.0
            y_origin = (
                b_next
                if not np.isnan(b_next) and abs(b_next) < (0.75 * hi)
                else 0.0
            )
            ax.quiver(
                0,
                y_origin,
                w_next[1] * arrow_scale,
                w_next[2] * arrow_scale,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#0F172A",
                width=0.011,
                label="Normal [w₁, w₂]ᵀ",
                zorder=6,
            )

    is_dense = (len(y_train) + len(y_test)) > 35
    pt_size = 62 if is_dense else 125
    pt_alpha = 0.85 if is_dense else 0.95
    mis_size = 135 if is_dense else 285

    if show_train_pts:
        tr_pos = y_train == 1.0
        tr_neg = y_train == 0.0
        ax.scatter(
            X_train[tr_pos, 0],
            X_train[tr_pos, 1],
            color="#2563EB",
            edgecolor="white",
            s=pt_size,
            alpha=pt_alpha,
            marker="o",
            linewidth=1.0 if is_dense else 1.4,
            label=f"Train y=1 (n={int(np.sum(tr_pos))})",
            zorder=5,
        )
        ax.scatter(
            X_train[tr_neg, 0],
            X_train[tr_neg, 1],
            color="#E11D48",
            edgecolor="white",
            s=pt_size,
            alpha=pt_alpha,
            marker="o",
            linewidth=1.0 if is_dense else 1.4,
            label=f"Train y=0 (n={int(np.sum(tr_neg))})",
            zorder=5,
        )

    if show_test_pts:
        te_pos = y_test == 1.0
        te_neg = y_test == 0.0
        ax.scatter(
            X_test[te_pos, 0],
            X_test[te_pos, 1],
            color="#60A5FA",
            edgecolor="#1E3A8A",
            s=pt_size,
            alpha=pt_alpha,
            marker="s",
            linewidth=1.1 if is_dense else 1.6,
            label=f"Test y=1 (n={int(np.sum(te_pos))})",
            zorder=6,
        )
        ax.scatter(
            X_test[te_neg, 0],
            X_test[te_neg, 1],
            color="#FB7185",
            edgecolor="#881337",
            s=pt_size,
            alpha=pt_alpha,
            marker="s",
            linewidth=1.1 if is_dense else 1.6,
            label=f"Test y=0 (n={int(np.sum(te_neg))})",
            zorder=6,
        )

    tr_mis = (preds_tr_next >= tau).astype(float) != y_train
    te_mis = (preds_te_next >= tau).astype(float) != y_test

    mis_coords = []
    if show_train_pts and np.any(tr_mis):
        mis_coords.append(X_train[tr_mis])
    if show_test_pts and np.any(te_mis):
        mis_coords.append(X_test[te_mis])

    if mis_coords:
        all_mis = np.vstack(mis_coords)
        ax.scatter(
            all_mis[:, 0],
            all_mis[:, 1],
            facecolors="none",
            edgecolors="#D97706",
            s=mis_size,
            linewidth=1.6 if is_dense else 2.0,
            linestyle="--",
            label=f"Misclassified (τ={tau:.2f})",
            zorder=7,
        )

    if show_pt_labels and not is_dense:
        if show_train_pts:
            for idx, pt in enumerate(X_train, start=1):
                ax.annotate(
                    f"tr{idx}",
                    xy=(pt[0], pt[1]),
                    xytext=(pt[0] + 0.09, pt[1] + 0.08),
                    fontsize=7.4,
                    fontweight="bold",
                    color="#0F172A",
                    zorder=8,
                )
        if show_test_pts:
            for idx, pt in enumerate(X_test, start=1):
                ax.annotate(
                    f"te{idx}",
                    xy=(pt[0], pt[1]),
                    xytext=(pt[0] + 0.09, pt[1] - 0.15),
                    fontsize=7.4,
                    fontweight="bold",
                    color="#334155",
                    zorder=8,
                )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("x₁", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax.set_ylabel("x₂", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax.tick_params(axis="both", which="major", labelsize=8.8, colors="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#CBD5E1")
    ax.spines["bottom"].set_edgecolor("#CBD5E1")
    ax.grid(True, linestyle=":", color="#CBD5E1", alpha=0.65)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=7.3,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=256, show_spinner=False)
def render_roc_png(
    fpr_tr: np.ndarray,
    tpr_tr: np.ndarray,
    auc_tr: float,
    fpr_op_tr: float,
    tpr_op_tr: float,
    rec_tr: float,
    fpr_te: np.ndarray,
    tpr_te: np.ndarray,
    auc_te: float,
    fpr_op_te: float,
    tpr_op_te: float,
    rec_te: float,
    tau: float,
    n_train: int,
) -> bytes:
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.15), dpi=150)
    fig_roc.patch.set_facecolor("#FFFFFF")
    ax_roc.set_facecolor("#F8FAFC")

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94A3B8",
        linewidth=1.4,
        label="Random Baseline (AUC = 0.500)",
        zorder=1,
    )

    mk_size = 3.2 if n_train > 30 else 4.5
    ax_roc.plot(
        fpr_tr,
        tpr_tr,
        color="#2563EB",
        linewidth=2.5,
        marker="o",
        markersize=mk_size,
        label=f"Train ROC (AUC = {auc_tr:.3f})",
        zorder=3,
    )
    ax_roc.fill_between(
        fpr_tr,
        tpr_tr,
        color="#2563EB",
        alpha=0.09,
        zorder=2,
    )

    ax_roc.plot(
        fpr_te,
        tpr_te,
        color="#059669",
        linewidth=2.4,
        linestyle="-.",
        marker="s",
        markersize=mk_size + 0.5,
        label=f"Test ROC (AUC = {auc_te:.3f})",
        zorder=4,
    )

    ax_roc.scatter(
        [fpr_op_tr],
        [tpr_op_tr],
        color="#1D4ED8",
        edgecolor="#D97706",
        s=140,
        linewidth=2.0,
        zorder=5,
        label=f"Train τ={tau:.2f} (TPR={rec_tr:.0f}%, FPR={fpr_op_tr*100:.0f}%)",
    )
    ax_roc.scatter(
        [fpr_op_te],
        [tpr_op_te],
        color="#047857",
        edgecolor="#D97706",
        s=140,
        marker="s",
        linewidth=2.0,
        zorder=6,
        label=f"Test τ={tau:.2f} (TPR={rec_te:.0f}%, FPR={fpr_op_te*100:.0f}%)",
    )

    ax_roc.set_xlim(-0.04, 1.04)
    ax_roc.set_ylim(-0.04, 1.06)
    ax_roc.set_xlabel(
        "False Positive Rate (FPR = 1 − Specificity)",
        fontsize=9.8,
        fontweight="bold",
        color="#1E293B",
    )
    ax_roc.set_ylabel(
        "True Positive Rate (TPR = Recall)",
        fontsize=9.8,
        fontweight="bold",
        color="#1E293B",
    )
    ax_roc.tick_params(axis="both", which="major", labelsize=8.8, colors="#334155")
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)
    ax_roc.spines["left"].set_edgecolor("#CBD5E1")
    ax_roc.spines["bottom"].set_edgecolor("#CBD5E1")
    ax_roc.grid(True, linestyle=":", color="#CBD5E1", alpha=0.75)
    ax_roc.set_aspect("equal", adjustable="box")

    ax_roc.legend(
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
    )
    fig_roc.tight_layout()
    buf = io.BytesIO()
    fig_roc.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig_roc.get_facecolor())
    plt.close(fig_roc)
    return buf.getvalue()


@st.cache_data(max_entries=256, show_spinner=False)
def render_pr_png(
    rec_curve_tr: np.ndarray,
    prec_curve_tr: np.ndarray,
    ap_tr: float,
    tpr_op_tr: float,
    prec_op_tr: float,
    prec_tr: float,
    rec_tr: float,
    rec_curve_te: np.ndarray,
    prec_curve_te: np.ndarray,
    ap_te: float,
    tpr_op_te: float,
    prec_op_te: float,
    prec_te: float,
    rec_te: float,
    pos_prior: float,
    tau: float,
    n_train: int,
) -> bytes:
    fig_pr, ax_pr = plt.subplots(figsize=(5.5, 4.15), dpi=150)
    fig_pr.patch.set_facecolor("#FFFFFF")
    ax_pr.set_facecolor("#F8FAFC")

    for f1_iso in [0.2, 0.4, 0.6, 0.8]:
        r_grid = np.linspace(f1_iso / (2.0 - f1_iso) + 1e-4, 1.0, 200)
        p_grid = (f1_iso * r_grid) / (2.0 * r_grid - f1_iso)
        valid = (p_grid >= 0.0) & (p_grid <= 1.0)
        ax_pr.plot(
            r_grid[valid],
            p_grid[valid],
            color="#CBD5E1",
            linestyle=":",
            linewidth=1.1,
            zorder=1,
        )
        r_lbl = 0.96
        p_lbl = (f1_iso * r_lbl) / (2.0 * r_lbl - f1_iso)
        if 0.05 < p_lbl < 0.95:
            ax_pr.annotate(
                f"F₁={f1_iso:.1f}",
                xy=(r_lbl, p_lbl),
                fontsize=6.9,
                color="#94A3B8",
                ha="right",
                va="bottom",
                zorder=1,
            )

    ax_pr.axhline(
        pos_prior,
        linestyle="--",
        color="#94A3B8",
        linewidth=1.4,
        label=f"No-Skill Baseline (P(y=1) = {pos_prior:.2f})",
        zorder=2,
    )

    mk_size = 3.2 if n_train > 30 else 4.5
    ax_pr.plot(
        rec_curve_tr,
        prec_curve_tr,
        color="#2563EB",
        linewidth=2.5,
        drawstyle="steps-pre",
        marker="o",
        markersize=mk_size,
        label=f"Train PR (AP = {ap_tr:.3f})",
        zorder=4,
    )
    ax_pr.fill_between(
        rec_curve_tr,
        prec_curve_tr,
        step="pre",
        color="#2563EB",
        alpha=0.09,
        zorder=3,
    )

    ax_pr.plot(
        rec_curve_te,
        prec_curve_te,
        color="#059669",
        linewidth=2.4,
        linestyle="-.",
        drawstyle="steps-pre",
        marker="s",
        markersize=mk_size + 0.5,
        label=f"Test PR (AP = {ap_te:.3f})",
        zorder=5,
    )

    ax_pr.scatter(
        [tpr_op_tr],
        [prec_op_tr],
        color="#1D4ED8",
        edgecolor="#D97706",
        s=140,
        linewidth=2.0,
        zorder=6,
        label=f"Train τ={tau:.2f} (P={prec_tr:.0f}%, R={rec_tr:.0f}%)",
    )
    ax_pr.scatter(
        [tpr_op_te],
        [prec_op_te],
        color="#047857",
        edgecolor="#D97706",
        s=140,
        marker="s",
        linewidth=2.0,
        zorder=7,
        label=f"Test τ={tau:.2f} (P={prec_te:.0f}%, R={rec_te:.0f}%)",
    )

    ax_pr.set_xlim(-0.04, 1.04)
    ax_pr.set_ylim(-0.04, 1.06)
    ax_pr.set_xlabel(
        "Recall / Sensitivity (TPR)",
        fontsize=9.8,
        fontweight="bold",
        color="#1E293B",
    )
    ax_pr.set_ylabel(
        "Precision / Positive Predictive Value (PPV)",
        fontsize=9.8,
        fontweight="bold",
        color="#1E293B",
    )
    ax_pr.tick_params(axis="both", which="major", labelsize=8.8, colors="#334155")
    ax_pr.spines["top"].set_visible(False)
    ax_pr.spines["right"].set_visible(False)
    ax_pr.spines["left"].set_edgecolor("#CBD5E1")
    ax_pr.spines["bottom"].set_edgecolor("#CBD5E1")
    ax_pr.grid(True, linestyle=":", color="#CBD5E1", alpha=0.75)
    ax_pr.set_aspect("equal", adjustable="box")

    ax_pr.legend(
        loc="lower left",
        fontsize=7.3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
    )
    fig_pr.tight_layout()
    buf = io.BytesIO()
    fig_pr.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig_pr.get_facecolor())
    plt.close(fig_pr)
    return buf.getvalue()


def render_evaluation_plots(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    state: dict,
    tau: float,
    logit_tau_val: float,
    mode: str,
    show_train_pts: bool,
    show_test_pts: bool,
    show_prob_shading: bool,
    show_pt_labels: bool,
    show_normal_arrow: bool,
    plot_bounds=(-2.9, 2.9),
):
    t_step = state["t"]
    w_next = state["w_next"]
    m_next = state["m_next"]
    b_next = state["b_next"]
    b_next_05 = state["b_next_05"]
    m_tr_next = state["metrics_tr_next"]
    m_te_next = state["metrics_te_next"]

    plot_tab_boundary, plot_tab_roc, plot_tab_pr = st.tabs(
        [
            f"Decision Boundary (τ={tau:.2f})",
            "ROC Curve (AUC)",
            "Precision–Recall (AP)",
        ]
    )

    with plot_tab_boundary:
        boundary_png = render_boundary_png(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            preds_tr_next=state["preds_tr_next"],
            preds_te_next=state["preds_te_next"],
            t_step=t_step,
            w_next_tuple=tuple(float(v) for v in w_next),
            m_next=m_next,
            b_next=b_next,
            b_next_05=b_next_05,
            tau=tau,
            logit_tau_val=logit_tau_val,
            mode=mode,
            show_train_pts=show_train_pts,
            show_test_pts=show_test_pts,
            show_prob_shading=show_prob_shading,
            show_pt_labels=show_pt_labels,
            show_normal_arrow=show_normal_arrow,
            plot_bounds=plot_bounds,
        )
        st.image(boundary_png, use_container_width=True)
        st.markdown(
            build_boundary_interpretation_html(
                t_step,
                m_next,
                b_next,
                b_next_05,
                tau,
                logit_tau_val,
                m_tr_next,
                m_te_next,
                len(y_train),
                len(y_test),
                mode,
            ),
            unsafe_allow_html=True,
        )

    with plot_tab_roc:
        roc_png = render_roc_png(
            fpr_tr=m_tr_next["fpr_curve"],
            tpr_tr=m_tr_next["tpr_curve"],
            auc_tr=m_tr_next["auc"],
            fpr_op_tr=m_tr_next["fpr_op"],
            tpr_op_tr=m_tr_next["tpr_op"],
            rec_tr=m_tr_next["rec"],
            fpr_te=m_te_next["fpr_curve"],
            tpr_te=m_te_next["tpr_curve"],
            auc_te=m_te_next["auc"],
            fpr_op_te=m_te_next["fpr_op"],
            tpr_op_te=m_te_next["tpr_op"],
            rec_te=m_te_next["rec"],
            tau=tau,
            n_train=len(y_train),
        )
        st.image(roc_png, use_container_width=True)
        st.markdown(
            build_roc_interpretation_html(t_step, m_tr_next, m_te_next, tau, mode),
            unsafe_allow_html=True,
        )

    with plot_tab_pr:
        pr_png = render_pr_png(
            rec_curve_tr=m_tr_next["rec_curve"],
            prec_curve_tr=m_tr_next["prec_curve"],
            ap_tr=m_tr_next["ap"],
            tpr_op_tr=m_tr_next["tpr_op"],
            prec_op_tr=m_tr_next["prec_op"],
            prec_tr=m_tr_next["prec"],
            rec_tr=m_tr_next["rec"],
            rec_curve_te=m_te_next["rec_curve"],
            prec_curve_te=m_te_next["prec_curve"],
            ap_te=m_te_next["ap"],
            tpr_op_te=m_te_next["tpr_op"],
            prec_op_te=m_te_next["prec_op"],
            prec_te=m_te_next["prec"],
            rec_te=m_te_next["rec"],
            pos_prior=m_tr_next["pos_prior"],
            tau=tau,
            n_train=len(y_train),
        )
        st.image(pr_png, use_container_width=True)
        st.markdown(
            build_pr_interpretation_html(t_step, m_tr_next, m_te_next, tau, mode),
            unsafe_allow_html=True,
        )


def render_kpi_strip(
    t: int,
    tau: float,
    m_t: float,
    m_next: float,
    b_t: float,
    b_next: float,
    m_tr_t: dict,
    m_tr_next: dict,
    m_te_t: dict,
    m_te_next: dict,
):
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8, kpi9 = st.columns(9, gap="small")
    kpi1.metric(f"Step (τ={tau:.2f})", f"t={t}→{t+1}")

    slope_delta_str = (
        fmt_delta(m_next - m_t, 3)
        if not np.isnan(m_t) and not np.isnan(m_next)
        else "Initial"
    )
    kpi2.metric(
        f"Slope m at w^({t+1})",
        f"{m_next:.3f}" if not np.isnan(m_next) else "Undef",
        delta=slope_delta_str,
        delta_color="off",
    )

    int_delta_str = (
        fmt_delta(b_next - b_t, 3)
        if not np.isnan(b_t) and not np.isnan(b_next)
        else "Initial"
    )
    kpi3.metric(
        f"Intercept b(τ={tau:.2f})",
        f"{b_next:.3f}" if not np.isnan(b_next) else "Undef",
        delta=int_delta_str,
        delta_color="off",
    )

    kpi4.metric(
        f"Train Acc (τ={tau:.2f})",
        f"{m_tr_next['acc']:.1f}%",
        delta=fmt_delta(m_tr_next["acc"] - m_tr_t["acc"], 1, "%"),
    )
    kpi5.metric(
        f"Test Acc (τ={tau:.2f})",
        f"{m_te_next['acc']:.1f}%",
        delta=fmt_delta(m_te_next["acc"] - m_te_t["acc"], 1, "%"),
    )
    kpi6.metric(
        f"Train F1 (τ={tau:.2f})",
        f"{m_tr_next['f1']:.1f}%",
        delta=fmt_delta(m_tr_next["f1"] - m_tr_t["f1"], 1, "%"),
    )
    kpi7.metric(
        f"Test F1 (τ={tau:.2f})",
        f"{m_te_next['f1']:.1f}%",
        delta=fmt_delta(m_te_next["f1"] - m_te_t["f1"], 1, "%"),
    )
    kpi8.metric(
        "Train ROC-AUC",
        f"{m_tr_next['auc']:.3f}",
        delta=fmt_delta(m_tr_next["auc"] - m_tr_t["auc"], 3),
    )
    kpi9.metric(
        "Test ROC-AUC",
        f"{m_te_next['auc']:.3f}",
        delta=fmt_delta(m_te_next["auc"] - m_te_t["auc"], 3),
    )


if "iteration" not in st.session_state:
    st.session_state.iteration = 0

if "sk_iteration" not in st.session_state:
    st.session_state.sk_iteration = 20

if "random_w0" not in st.session_state:
    st.session_state.random_w0 = sample_random_w0()


def step_forward():
    st.session_state.iteration += 1


def step_backward():
    st.session_state.iteration = max(0, st.session_state.iteration - 1)


def reset_iteration():
    st.session_state.iteration = 0


def reroll_random_w0():
    st.session_state.random_w0 = sample_random_w0()
    st.session_state.iteration = 0


def sk_step_forward():
    st.session_state.sk_iteration += 1


def sk_step_backward():
    st.session_state.sk_iteration = max(0, st.session_state.sk_iteration - 1)


def sk_reset_iteration():
    st.session_state.sk_iteration = 0


def sk_converge_iteration():
    st.session_state.sk_iteration = 40


with st.sidebar:
    st.markdown("### Model & Threshold Controls")

    tau = st.slider(
        "Decision Threshold (τ)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Predict ŷ = 1 when σ(x̃ᵀw^(t+1)) ≥ τ, or equivalently x̃ᵀw^(t+1) ≥ ln(τ/(1−τ)). "
            "Varying τ translates the decision boundary in parallel and shifts the "
            "confusion matrix / ROC / PR operating point without altering gradient updates."
        ),
    )
    logit_tau_val = logit_threshold(tau)
    st.caption(f"Cutoff: **ŷ ≥ {tau:.2f}** ⟺ log-odds **aₙ ≥ {logit_tau_val:.3f}**")

    eta = st.slider(
        "Learning Rate (η)",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.01,
    )

    init_mode = st.selectbox(
        "Initial Weights w^(0)",
        options=[
            "Random Initialization (w^(0) ~ Uniform)",
            "Origin [0.0, 0.0, 0.0]ᵀ",
            "Off-Center [-0.4, 0.8, 0.3]ᵀ",
            "Custom Initial Weights",
        ],
        index=0,
        on_change=reset_iteration,
    )

    if init_mode == "Random Initialization (w^(0) ~ Uniform)":
        st.button(
            "Re-Sample Random w^(0)",
            on_click=reroll_random_w0,
            use_container_width=True,
        )
        rw = st.session_state.random_w0
        st.caption(f"Active **w^(0)** = `[{rw[0]:.2f}, {rw[1]:.2f}, {rw[2]:.2f}]ᵀ`")
        w_init_vec = rw.copy()
    elif init_mode == "Origin [0.0, 0.0, 0.0]ᵀ":
        w_init_vec = np.array([0.0, 0.0, 0.0], dtype=float)
    elif init_mode == "Off-Center [-0.4, 0.8, 0.3]ᵀ":
        w_init_vec = np.array([-0.4, 0.8, 0.3], dtype=float)
    else:
        c1, c2, c3 = st.columns(3)
        w0_bias = c1.number_input("w_bias^(0)", value=-0.4, step=0.1, format="%.2f")
        w0_1 = c2.number_input("w₁^(0)", value=0.8, step=0.1, format="%.2f")
        w0_2 = c3.number_input("w₂^(0)", value=0.3, step=0.1, format="%.2f")
        w_init_vec = np.array([w0_bias, w0_1, w0_2], dtype=float)

    w_init_tuple = tuple(float(v) for v in w_init_vec)

    st.divider()
    st.markdown("### Plot & Partition Overlays")
    split_view = st.radio(
        "Displayed Data Partition",
        options=[
            "Both Training & Testing",
            "Training Partition Only (Circles)",
            "Testing Partition Only (Squares)",
        ],
        index=0,
    )
    show_train_pts = split_view in (
        "Both Training & Testing",
        "Training Partition Only (Circles)",
    )
    show_test_pts = split_view in (
        "Both Training & Testing",
        "Testing Partition Only (Squares)",
    )

    show_prob_shading = st.checkbox("Show Soft Probability Shading", value=True)
    show_pt_labels = st.checkbox("Show Point IDs (Small N Only)", value=False)
    show_normal_arrow = st.checkbox("Show Normal Vector [w₁, w₂]ᵀ", value=False)


@st.fragment
def render_controlled_fragment(
    tau: float,
    logit_tau_val: float,
    eta: float,
    w_init_tuple: tuple,
    show_train_pts: bool,
    show_test_pts: bool,
    show_prob_shading: bool,
    show_pt_labels: bool,
    show_normal_arrow: bool,
):
    dataset_choice = st.selectbox(
        "Controlled 2D Geometry (70% Train [N = 14: 7 Pos, 7 Neg] | 30% Test [N = 6: 3 Pos, 3 Neg])",
        options=list(SYNTHETIC_DATASETS.keys()),
        index=1,
        on_change=reset_iteration,
    )

    ds = SYNTHETIC_DATASETS[dataset_choice]
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]
    mode = ds["mode"]

    X_train_tilde = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_tilde = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    state = get_step_state(
        X_train_tilde,
        y_train,
        X_test_tilde,
        y_test,
        w_init_tuple,
        eta,
        st.session_state.iteration,
        tau=tau,
        normalize_grad=False,
    )

    t = state["t"]
    w_t = state["w_t"]
    w_next = state["w_next"]
    grad = state["grad"]
    m_t, b_t = state["m_t"], state["b_t"]
    m_next, b_next, b_next_05 = state["m_next"], state["b_next"], state["b_next_05"]
    m_tr_t, m_tr_next = state["metrics_tr_t"], state["metrics_tr_next"]
    m_te_t, m_te_next = state["metrics_te_t"], state["metrics_te_next"]

    render_kpi_strip(
        t, tau, m_t, m_next, b_t, b_next, m_tr_t, m_tr_next, m_te_t, m_te_next
    )

    col_left, col_right = st.columns([1.08, 1.0], gap="large")

    with col_left:
        st.markdown(
            f'<div class="section-label">1. Current Weights w<sup>({t})</sup> & Training Batch Gradient (t = {t}, N_train = 14 — Invariant to τ)</div>',
            unsafe_allow_html=True,
        )
        st.latex(
            rf"\bm{{w}}^{{({t})}} = \begin{{bmatrix}} w_{{\text{{bias}}}}^{{({t})}} \\ w_1^{{({t})}} \\ w_2^{{({t})}} \end{{bmatrix}} = "
            rf"\begin{{bmatrix}} {w_t[0]:.4f} \\ {w_t[1]:.4f} \\ {w_t[2]:.4f} \end{{bmatrix}}, \quad "
            rf"\nabla_{{\bm{{w}}}} \ell_{{\text{{train}}}}(\bm{{w}}^{{({t})}}) = \sum_{{n=1}}^{{14}} (\hat{{y}}_n - y_n)\tilde{{\bm{{x}}}}_n = "
            rf"\begin{{bmatrix}} {grad[0]:.4f} \\ {grad[1]:.4f} \\ {grad[2]:.4f} \end{{bmatrix}}"
        )

        st.markdown(
            f'<div class="section-label">2. Parameter Update w<sup>({t+1})</sup> & Threshold Boundary (τ = {tau:.2f} ⟺ logit(τ) = {logit_tau_val:.3f})</div>',
            unsafe_allow_html=True,
        )
        st.latex(
            rf"\bm{{w}}^{{({t+1})}} = \bm{{w}}^{{({t})}} - \eta \nabla_{{\bm{{w}}}} \ell_{{\text{{train}}}}(\bm{{w}}^{{({t})}}) = "
            rf"\begin{{bmatrix}} {w_t[0]:.4f} \\ {w_t[1]:.4f} \\ {w_t[2]:.4f} \end{{bmatrix}} - "
            rf"{eta:.2f} \begin{{bmatrix}} {grad[0]:.4f} \\ {grad[1]:.4f} \\ {grad[2]:.4f} \end{{bmatrix}} = "
            rf"\begin{{bmatrix}} \mathbf{{{w_next[0]:.4f}}} \\ \mathbf{{{w_next[1]:.4f}}} \\ \mathbf{{{w_next[2]:.4f}}} \end{{bmatrix}}"
        )
        st.latex(
            rf"\hat{{y}} \ge {tau:.2f} \iff \tilde{{\bm{{x}}}}^\top \bm{{w}}^{{({t+1})}} \ge \ln\!\left(\frac{{{tau:.2f}}}{{{1.0-tau:.2f}}}\right) = {logit_tau_val:.3f} "
            rf"\implies x_2 = \underbrace{{\mathbf{{{m_next:.3f}}}}}_{{\text{{Slope }} m = -w_1^{{({t+1})}}/w_2^{{({t+1})}}}}\,x_1 + "
            rf"\underbrace{{{b_next:.3f}}}_{{\text{{Intercept }} b(\tau) = (\operatorname{{logit}}(\tau) - w_{{\text{{bias}}}}^{{({t+1})}})/w_2^{{({t+1})}}}}"
        )

        f1_gap = m_tr_next["f1"] - m_te_next["f1"]
        ap_gap = m_tr_next["ap"] - m_te_next["ap"]

        if mode == "overlapping":
            st.markdown(
                '<div class="status-banner">'
                f"<b>Boundary-Zone Overlap (τ = {tau:.2f}):</b> Train has 2/14 (14.3%) boundary crossovers (<b>tr<sub>7</sub>, tr<sub>8</sub></b>) "
                f"and Test has 1/6 (16.7%) crossover (<b>te<sub>4</sub></b>). At <b>w<sup>({t+1})</sup></b> and <b>τ = {tau:.2f}</b>: <b>Train F1 = {m_tr_next['f1']:.1f}%</b>, <b>Test F1 = {m_te_next['f1']:.1f}%</b> "
                f"(Test Acc = <b>{m_te_next['acc']:.1f}%</b>, Test ROC-AUC = <b>{m_te_next['auc']:.3f}</b>, Test PR-AUC = <b>{m_te_next['ap']:.3f}</b>)."
                "</div>",
                unsafe_allow_html=True,
            )
        elif mode == "overfitting":
            st.markdown(
                '<div class="status-banner">'
                "<b>Overfitting Regime:</b> Margin outliers <b>tr<sub>6</sub>, tr<sub>7</sub></b> (Q4) and <b>tr<sub>8</sub>, tr<sub>9</sub></b> (Q2) "
                f"rotate the boundary away from <i>x</i><sub>2</sub> = 0. At <b>w<sup>({t+1})</sup></b> and <b>τ = {tau:.2f}</b>, "
                f"Train Acc is <b>{m_tr_next['acc']:.1f}%</b> (F1 <b>{m_tr_next['f1']:.1f}%</b>) while Test Acc is "
                f"<b>{m_te_next['acc']:.1f}%</b> (F1 <b>{m_te_next['f1']:.1f}%</b>, Gap = <b>{f1_gap:.1f}%</b>)."
                "</div>",
                unsafe_allow_html=True,
            )
        elif mode == "doughnut":
            st.markdown(
                '<div class="status-banner">'
                f"<b>Underfitting Regime (High Bias, τ = {tau:.2f}):</b> A linear boundary cannot enclose the concentric inner core. "
                f"At <b>w<sup>({t+1})</sup></b>, both splits plateau near random guessing: Train Acc = <b>{m_tr_next['acc']:.1f}%</b> (ROC-AUC <b>{m_tr_next['auc']:.3f}</b>) "
                f"and Test Acc = <b>{m_te_next['acc']:.1f}%</b> (ROC-AUC <b>{m_te_next['auc']:.3f}</b>)."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-banner">'
                f"<b>Linearly Separable Generalization (τ = {tau:.2f}):</b> At <b>w<sup>({t+1})</sup></b>, "
                f"Train Acc = <b>{m_tr_next['acc']:.1f}%</b> (F1 <b>{m_tr_next['f1']:.1f}%</b>) and "
                f"Test Acc = <b>{m_te_next['acc']:.1f}%</b> (F1 <b>{m_te_next['f1']:.1f}%</b>) | "
                f"Gap (Train − Test) F1 = <b>{f1_gap:.1f}%</b>, PR-AUC = <b>{ap_gap:.3f}</b>."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="section-label">3. Classification Metrics (at w<sup>({t+1})</sup>, τ = {tau:.2f}) & Per-Sample Forward Pass (at w<sup>({t})</sup>)</div>',
            unsafe_allow_html=True,
        )

        tab_metrics, tab_tr, tab_te = st.tabs(
            [
                f"Metrics & Confusion Matrix (τ={tau:.2f})",
                "Train Samples (N = 14 | 70%)",
                "Test Samples (N = 6 | 30%)",
            ]
        )

        with tab_metrics:
            st.markdown(
                build_metrics_comparison_html(
                    m_tr_t, m_tr_next, m_te_t, m_te_next, t, tau, 14, 6
                ),
                unsafe_allow_html=True,
            )
            cm_col1, cm_col2 = st.columns(2, gap="small")
            with cm_col1:
                st.markdown(
                    build_confusion_matrix_html(
                        m_tr_next,
                        f"Train Confusion Matrix (w<sup>({t+1})</sup>, τ={tau:.2f})",
                        tau=tau,
                    ),
                    unsafe_allow_html=True,
                )
            with cm_col2:
                st.markdown(
                    build_confusion_matrix_html(
                        m_te_next,
                        f"Test Confusion Matrix (w<sup>({t+1})</sup>, τ={tau:.2f})",
                        tau=tau,
                    ),
                    unsafe_allow_html=True,
                )

        with tab_tr:
            tcol1, tcol2 = st.columns(2, gap="small")
            with tcol1:
                st.markdown(
                    build_sample_table_html(
                        X_train[:7],
                        y_train[:7],
                        state["logits_tr"][:7],
                        state["preds_tr"][:7],
                        state["errors_tr"][:7],
                        w_t,
                        t_step=t,
                        tau=tau,
                        prefix="tr",
                        start_idx=1,
                    ),
                    unsafe_allow_html=True,
                )
            with tcol2:
                st.markdown(
                    build_sample_table_html(
                        X_train[7:],
                        y_train[7:],
                        state["logits_tr"][7:],
                        state["preds_tr"][7:],
                        state["errors_tr"][7:],
                        w_t,
                        t_step=t,
                        tau=tau,
                        prefix="tr",
                        start_idx=8,
                    ),
                    unsafe_allow_html=True,
                )

        with tab_te:
            te_col1, te_col2 = st.columns(2, gap="small")
            with te_col1:
                st.markdown(
                    build_sample_table_html(
                        X_test[:3],
                        y_test[:3],
                        state["logits_te"][:3],
                        state["preds_te"][:3],
                        state["errors_te"][:3],
                        w_t,
                        t_step=t,
                        tau=tau,
                        prefix="te",
                        start_idx=1,
                    ),
                    unsafe_allow_html=True,
                )
            with te_col2:
                st.markdown(
                    build_sample_table_html(
                        X_test[3:],
                        y_test[3:],
                        state["logits_te"][3:],
                        state["preds_te"][3:],
                        state["errors_te"][3:],
                        w_t,
                        t_step=t,
                        tau=tau,
                        prefix="te",
                        start_idx=4,
                    ),
                    unsafe_allow_html=True,
                )

    with col_right:
        render_evaluation_plots(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            state=state,
            tau=tau,
            logit_tau_val=logit_tau_val,
            mode=mode,
            show_train_pts=show_train_pts,
            show_test_pts=show_test_pts,
            show_prob_shading=show_prob_shading,
            show_pt_labels=show_pt_labels,
            show_normal_arrow=show_normal_arrow,
            plot_bounds=(-2.9, 2.9),
        )

        st.markdown(
            f'<div class="section-label">Iteration Controller — Step t = {t} → {t+1} (w<sup>({t})</sup> → w<sup>({t+1})</sup>, η = {eta:.2f}, τ = {tau:.2f})</div>',
            unsafe_allow_html=True,
        )
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.0, 1.0, 1.15], gap="small")
        with ctrl_col1:
            st.button(
                "Previous Step",
                on_click=step_backward,
                disabled=(st.session_state.iteration == 0),
                use_container_width=True,
                key="syn_prev",
            )
        with ctrl_col2:
            st.button(
                "Next Step",
                on_click=step_forward,
                type="primary",
                use_container_width=True,
                key="syn_next",
            )
        with ctrl_col3:
            st.button(
                "Reset (t = 0)",
                on_click=reset_iteration,
                disabled=(st.session_state.iteration == 0),
                use_container_width=True,
                key="syn_reset",
            )


@st.fragment
def render_sklearn_fragment(
    tau: float,
    logit_tau_val: float,
    eta: float,
    w_init_tuple: tuple,
    show_train_pts: bool,
    show_test_pts: bool,
    show_prob_shading: bool,
    show_pt_labels: bool,
    show_normal_arrow: bool,
):
    cfg1, cfg2, cfg3, cfg4, cfg5 = st.columns(
        [1.55, 0.85, 0.85, 0.85, 0.75], gap="small"
    )
    with cfg1:
        sk_gen_choice = st.selectbox(
            "Sklearn 2D Dataset Generator",
            options=[
                "1. Two Interleaving Moons (make_moons)",
                "2. Concentric Circles (make_circles)",
                "3. Isotropic Gaussian Blobs (make_blobs)",
                "4. Correlated Gaussian Clusters (make_classification)",
            ],
            index=0,
        )
    with cfg2:
        sk_n_samples = st.slider(
            "Total Samples (N)",
            min_value=40,
            max_value=240,
            value=120,
            step=20,
        )
    with cfg3:
        sk_noise = st.slider(
            "Noise / Dispersion",
            min_value=0.05,
            max_value=0.40,
            value=0.15,
            step=0.05,
        )
    with cfg4:
        sk_test_pct = st.slider(
            "Test Split (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
        )
    with cfg5:
        sk_seed = int(
            st.number_input(
                "Random Seed",
                min_value=0,
                max_value=99999,
                value=42,
                step=1,
            )
        )

    sk_data = generate_sklearn_2d_split(
        sk_gen_choice,
        sk_n_samples,
        sk_noise,
        sk_test_pct / 100.0,
        sk_seed,
    )
    X_tr_sk = sk_data["X_train"]
    y_tr_sk = sk_data["y_train"]
    X_te_sk = sk_data["X_test"]
    y_te_sk = sk_data["y_test"]
    X_tr_sk_tilde = sk_data["X_train_tilde"]
    X_te_sk_tilde = sk_data["X_test_tilde"]
    sk_mode = sk_data["mode"]
    n_tr_sk = len(y_tr_sk)
    n_te_sk = len(y_te_sk)

    sk_eta = eta * 2.5
    sk_state = get_step_state(
        X_tr_sk_tilde,
        y_tr_sk,
        X_te_sk_tilde,
        y_te_sk,
        w_init_tuple,
        sk_eta,
        st.session_state.sk_iteration,
        tau=tau,
        normalize_grad=True,
    )

    sk_t = sk_state["t"]
    sk_w_t = sk_state["w_t"]
    sk_grad = sk_state["grad"]
    sk_m_t, sk_b_t = sk_state["m_t"], sk_state["b_t"]
    sk_m_next, sk_b_next = sk_state["m_next"], sk_state["b_next"]
    sk_m_tr_t, sk_m_tr_next = sk_state["metrics_tr_t"], sk_state["metrics_tr_next"]
    sk_m_te_t, sk_m_te_next = sk_state["metrics_te_t"], sk_state["metrics_te_next"]

    render_kpi_strip(
        sk_t,
        tau,
        sk_m_t,
        sk_m_next,
        sk_b_t,
        sk_b_next,
        sk_m_tr_t,
        sk_m_tr_next,
        sk_m_te_t,
        sk_m_te_next,
    )

    sk_left, sk_right = st.columns([1.08, 1.0], gap="large")

    with sk_left:
        st.markdown(
            f'<div class="section-label">1. Current Weights w<sup>({sk_t})</sup> & Mean Training Gradient (t = {sk_t}, N_train = {n_tr_sk} — Invariant to τ)</div>',
            unsafe_allow_html=True,
        )
        st.latex(
            rf"\bm{{w}}^{{({sk_t})}} = \begin{{bmatrix}} w_{{\text{{bias}}}}^{{({sk_t})}} \\ w_1^{{({sk_t})}} \\ w_2^{{({sk_t})}} \end{{bmatrix}} = "
            rf"\begin{{bmatrix}} {sk_w_t[0]:.4f} \\ {sk_w_t[1]:.4f} \\ {sk_w_t[2]:.4f} \end{{bmatrix}}, \quad "
            rf"\frac{{1}}{{{n_tr_sk}}}\nabla_{{\bm{{w}}}} \ell_{{\text{{train}}}}(\bm{{w}}^{{({sk_t})}}) = "
            rf"\frac{{1}}{{{n_tr_sk}}}\sum_{{n=1}}^{{{n_tr_sk}}} (\hat{{y}}_n - y_n)\tilde{{\bm{{x}}}}_n = "
            rf"\begin{{bmatrix}} {sk_grad[0]:.4f} \\ {sk_grad[1]:.4f} \\ {sk_grad[2]:.4f} \end{{bmatrix}}"
        )

        st.markdown(
            f'<div class="section-label">2. Parameter Update w<sup>({sk_t+1})</sup> & Decision Boundary in (x₁, x₂) (τ = {tau:.2f} ⟺ logit(τ) = {logit_tau_val:.3f})</div>',
            unsafe_allow_html=True,
        )
        st.latex(
            rf"\hat{{y}} \ge {tau:.2f} \iff w_{{\text{{bias}}}}^{{({sk_t+1})}} + w_1^{{({sk_t+1})}} x_1 + w_2^{{({sk_t+1})}} x_2 \ge {logit_tau_val:.3f} "
            rf"\implies x_2 = \underbrace{{\mathbf{{{sk_m_next:.3f}}}}}_{{\text{{Slope }} m = -w_1^{{({sk_t+1})}}/w_2^{{({sk_t+1})}}}}\,x_1 + "
            rf"\underbrace{{{sk_b_next:.3f}}}_{{\text{{Intercept }} b(\tau) = (\operatorname{{logit}}(\tau) - w_{{\text{{bias}}}}^{{({sk_t+1})}})/w_2^{{({sk_t+1})}}}}"
        )

        st.markdown(
            f'<div class="status-banner">'
            f"<b>Generator Summary (Seed = {sk_seed}):</b> Stratified <b>{100 - sk_test_pct}% Train (N = {n_tr_sk})</b> / "
            f"<b>{sk_test_pct}% Test (N = {n_te_sk})</b> with noise = <b>{sk_noise:.2f}</b>. {sk_data['description']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="section-label">3. Evaluation Metrics (at w<sup>({sk_t+1})</sup>, τ = {tau:.2f}) & Sample Forward Pass (at w<sup>({sk_t})</sup>)</div>',
            unsafe_allow_html=True,
        )

        sk_tab_metrics, sk_tab_tr, sk_tab_te = st.tabs(
            [
                f"Metrics & Confusion Matrix (τ={tau:.2f})",
                f"Train Subset (First 14 of {n_tr_sk})",
                f"Test Subset (First 14 of {n_te_sk})",
            ]
        )

        with sk_tab_metrics:
            st.markdown(
                build_metrics_comparison_html(
                    sk_m_tr_t,
                    sk_m_tr_next,
                    sk_m_te_t,
                    sk_m_te_next,
                    sk_t,
                    tau,
                    n_tr_sk,
                    n_te_sk,
                ),
                unsafe_allow_html=True,
            )
            sk_cm1, sk_cm2 = st.columns(2, gap="small")
            with sk_cm1:
                st.markdown(
                    build_confusion_matrix_html(
                        sk_m_tr_next,
                        f"Train Confusion Matrix (w<sup>({sk_t+1})</sup>, N={n_tr_sk}, τ={tau:.2f})",
                        tau=tau,
                    ),
                    unsafe_allow_html=True,
                )
            with sk_cm2:
                st.markdown(
                    build_confusion_matrix_html(
                        sk_m_te_next,
                        f"Test Confusion Matrix (w<sup>({sk_t+1})</sup>, N={n_te_sk}, τ={tau:.2f})",
                        tau=tau,
                    ),
                    unsafe_allow_html=True,
                )

        with sk_tab_tr:
            n_show_tr = min(14, n_tr_sk)
            half_tr = (n_show_tr + 1) // 2
            sk_tcol1, sk_tcol2 = st.columns(2, gap="small")
            with sk_tcol1:
                st.markdown(
                    build_sample_table_html(
                        X_tr_sk[:half_tr],
                        y_tr_sk[:half_tr],
                        sk_state["logits_tr"][:half_tr],
                        sk_state["preds_tr"][:half_tr],
                        sk_state["errors_tr"][:half_tr],
                        sk_w_t,
                        t_step=sk_t,
                        tau=tau,
                        prefix="tr",
                        start_idx=1,
                    ),
                    unsafe_allow_html=True,
                )
            with sk_tcol2:
                st.markdown(
                    build_sample_table_html(
                        X_tr_sk[half_tr:n_show_tr],
                        y_tr_sk[half_tr:n_show_tr],
                        sk_state["logits_tr"][half_tr:n_show_tr],
                        sk_state["preds_tr"][half_tr:n_show_tr],
                        sk_state["errors_tr"][half_tr:n_show_tr],
                        sk_w_t,
                        t_step=sk_t,
                        tau=tau,
                        prefix="tr",
                        start_idx=half_tr + 1,
                    ),
                    unsafe_allow_html=True,
                )

        with sk_tab_te:
            n_show_te = min(14, n_te_sk)
            half_te = (n_show_te + 1) // 2
            sk_tecol1, sk_tecol2 = st.columns(2, gap="small")
            with sk_tecol1:
                st.markdown(
                    build_sample_table_html(
                        X_te_sk[:half_te],
                        y_te_sk[:half_te],
                        sk_state["logits_te"][:half_te],
                        sk_state["preds_te"][:half_te],
                        sk_state["errors_te"][:half_te],
                        sk_w_t,
                        t_step=sk_t,
                        tau=tau,
                        prefix="te",
                        start_idx=1,
                    ),
                    unsafe_allow_html=True,
                )
            with sk_tecol2:
                if n_show_te > half_te:
                    st.markdown(
                        build_sample_table_html(
                            X_te_sk[half_te:n_show_te],
                            y_te_sk[half_te:n_show_te],
                            sk_state["logits_te"][half_te:n_show_te],
                            sk_state["preds_te"][half_te:n_show_te],
                            sk_state["errors_te"][half_te:n_show_te],
                            sk_w_t,
                            t_step=sk_t,
                            tau=tau,
                            prefix="te",
                            start_idx=half_te + 1,
                        ),
                        unsafe_allow_html=True,
                    )

    with sk_right:
        max_extent = float(
            max(np.max(np.abs(X_tr_sk)), np.max(np.abs(X_te_sk)), 2.4) * 1.12
        )
        render_evaluation_plots(
            X_train=X_tr_sk,
            y_train=y_tr_sk,
            X_test=X_te_sk,
            y_test=y_te_sk,
            state=sk_state,
            tau=tau,
            logit_tau_val=logit_tau_val,
            mode=sk_mode,
            show_train_pts=show_train_pts,
            show_test_pts=show_test_pts,
            show_prob_shading=show_prob_shading,
            show_pt_labels=show_pt_labels,
            show_normal_arrow=show_normal_arrow,
            plot_bounds=(-max_extent, max_extent),
        )

        st.markdown(
            f'<div class="section-label">Iteration Controller — Step t = {sk_t} → {sk_t+1} (w<sup>({sk_t})</sup> → w<sup>({sk_t+1})</sup>, Effective η = {sk_eta:.2f}, τ = {tau:.2f})</div>',
            unsafe_allow_html=True,
        )
        sctrl1, sctrl2, sctrl3, sctrl4 = st.columns(4, gap="small")
        with sctrl1:
            st.button(
                "Previous Step",
                on_click=sk_step_backward,
                disabled=(st.session_state.sk_iteration == 0),
                use_container_width=True,
                key="sk_prev",
            )
        with sctrl2:
            st.button(
                "Next Step",
                on_click=sk_step_forward,
                type="primary",
                use_container_width=True,
                key="sk_next",
            )
        with sctrl3:
            st.button(
                "Jump to t = 40",
                on_click=sk_converge_iteration,
                use_container_width=True,
                key="sk_conv",
            )
        with sctrl4:
            st.button(
                "Reset (t = 0)",
                on_click=sk_reset_iteration,
                disabled=(st.session_state.sk_iteration == 0),
                use_container_width=True,
                key="sk_reset",
            )


st.markdown("# Logistic Regression Demo")
st.markdown(
    "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu",
)
main_tab_controlled, main_tab_sklearn = st.tabs(
    [
        "1. Controlled 2D Benchmarks (N = 20)",
        "2. Sklearn 2D Generators (Moons, Circles & Clusters)",
    ]
)

with main_tab_controlled:
    render_controlled_fragment(
        tau=tau,
        logit_tau_val=logit_tau_val,
        eta=eta,
        w_init_tuple=w_init_tuple,
        show_train_pts=show_train_pts,
        show_test_pts=show_test_pts,
        show_prob_shading=show_prob_shading,
        show_pt_labels=show_pt_labels,
        show_normal_arrow=show_normal_arrow,
    )

with main_tab_sklearn:
    render_sklearn_fragment(
        tau=tau,
        logit_tau_val=logit_tau_val,
        eta=eta,
        w_init_tuple=w_init_tuple,
        show_train_pts=show_train_pts,
        show_test_pts=show_test_pts,
        show_prob_shading=show_prob_shading,
        show_pt_labels=show_pt_labels,
        show_normal_arrow=show_normal_arrow,
    )
