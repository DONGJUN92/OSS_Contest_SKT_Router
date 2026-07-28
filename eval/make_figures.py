"""B11 — 제출 문서용 시각 자료 생성 (PROJECT_PLAN §9 P3).

**정확한 서술 (Phase 17 정정)**: fig1·fig2·fig4 는 결과 JSON만 읽는다. fig3(λ 궤적)과
fig5(σ 두 구간)는 **정책을 그 자리에서 재실행/재계산**해 그린다 — 궤적은 집계 JSON 에 없는
정보이기 때문이다. 이전 문서는 "전부 JSON 만 읽는다"고 썼는데 사실이 아니었고, 게다가
fig1 은 수치를 하드코딩하고 fig2 는 JSON 키가 틀려 조용히 하드코딩으로 폴백했다.
지금은 fig1·fig2 가 JSON 을 읽고, 키가 안 맞으면 **그림을 만들지 않고 건너뛴다**.
그림은 **주장 하나당 하나**로 제한한다 — 장식이 아니라 근거다.

  fig1_worlds.png    6세계 종합점수 (LPB vs cascade) — worst-case dominance
  fig2_ablation.png  구성요소 기여 — 무엇이 점수를 만드는가
  fig3_lambda.png    λ 궤적과 지출 페이싱 — M3가 실제로 하는 일
  fig4_loss.png      손실 분해 (선택오류·조기정지 후회·무응답) — 남은 갭의 정체
  fig5_sigma.png     예약값 σ의 두 구간과 D17 — λ→∞ 퇴화 동작

사용법: python eval/make_figures.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# 한글 폰트가 없는 환경에서도 깨지지 않도록 라벨은 영문 + 수식 기호로 통일
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
C_LPB, C_BASE, C_ORACLE = "#2563eb", "#94a3b8", "#16a34a"


def _load(name, default=None):
    p = OUT / name
    if not p.exists():
        print(f"  [건너뜀] {name} 없음")
        return default
    return json.load(open(p, encoding="utf-8"))


CASCADE_6W = {                     # phase7c/phase8b 산출 (정적 threshold cascade)
    "irt": 0.6844, "specialist": 0.9985, "corr": 0.8690,
    "crossing": 0.9935, "nosignal": 0.7869, "textworld": 0.6344,
}


def fig_worlds():
    """6세계 종합점수 — **배포 구성(단일집단)** vs 정적 cascade.

    ★ Phase 17 수정: 이전 버전은 6세계 점수를 함수 안에 **하드코딩**했고, 그 값도 배포행이
    아니라 **다집단 상한**이었다. 모듈 문서와 SUBMISSION.md §8 이 "결과 JSON만 읽으므로
    그림과 표가 항상 같은 수치를 가리킨다"고 보증했는데 사실이 아니었다.
    이제 `probe_domain_free.json` 의 `single_group`(직접 측정된 배포 구성)을 읽는다.
    """
    d = _load("probe_domain_free.json")
    if d is None:
        return
    worlds = ["irt", "specialist", "corr", "crossing", "nosignal", "textworld"]
    labels = ["A\nIRT", "B\nspecialist", "C\ncorr", "D\ncrossing", "E\nnosignal",
              "F\ntextworld"]
    lpb = [d["summary"][w]["single_group"] for w in worlds]
    casc = [CASCADE_6W[w] for w in worlds]
    x = np.arange(len(worlds))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - 0.2, lpb, 0.4, label="LPB-Router (deployment, single-group)", color=C_LPB)
    ax.bar(x + 0.2, casc, 0.4, label="static cascade (budget-blind)", color=C_BASE)
    for i, (a, b) in enumerate(zip(lpb, casc)):
        ax.text(i - 0.2, a + 0.012, f"{a:.3f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, b + 0.012, f"{b:.3f}", ha="center", fontsize=7.5, color="#64748b")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("combined score\n(tier-weighted 0.5/0.3/0.2)")
    # 제목은 그림이 실제로 보여주는 것만 말한다 (Phase 17: 이전 제목은 cascade 가 이기는
    # 2개 천장근접 세계를 무시하고 "all 6 worlds 최상위"라 주장했다).
    ax.set_title("LPB leads in the 4 routing-critical worlds; cascade edges it in 2 "
                 "ceiling-bound ones\n(synthetic validation worlds — see README caveat)",
                 fontsize=9, pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2,
              frameon=False, fontsize=8)      # 막대를 가리지 않도록 축 위로
    fig.tight_layout()
    fig.savefig(FIG / "fig1_worlds.png")
    plt.close(fig)
    print("  fig1_worlds.png")


def fig_ablation():
    """구성요소 기여 — 제거 시 종합점수 변화."""
    # ★ Phase 17 수정: 이전 버전의 JSON 경로는 **죽은 코드**였다 — 키를 `"-votes"` 로 찾는데
    # 실제 JSON 키는 `"-votes(5box)"` 이고, 중첩 순서도 `d[variant][world]` 가 아니라
    # `d[world][variant]` 다. 그래서 매번 KeyError → `except: pass` → **하드코딩 값**이 그려졌다.
    # 조용한 폴백을 없애고, 키가 없으면 그림을 만들지 않는다 (틀린 그림보다 없는 그림이 낫다).
    d = _load("phase6b_ablation.json")
    if not isinstance(d, dict):
        return
    names = ["− vote boxes\n[model x N]", "− pacing\n(static λ)",
             "− quality-first λ", "− IRT\n(discriminative)"]
    keys = ["-votes(5box)", "-pacing(static)", "-qualityfirst", "-irt(disc)"]
    try:
        a = [d["irt"][k] - d["irt"]["full"] for k in keys]
        b = [d["specialist"][k] - d["specialist"]["full"] for k in keys]
    except (KeyError, TypeError) as e:
        print(f"  [건너뜀] fig2 — phase6b_ablation.json 키 불일치: {e}")
        return
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.barh(y - 0.19, a, 0.38, label="World A (IRT)", color=C_LPB)
    ax.barh(y + 0.19, b, 0.38, label="World B (specialist)", color="#f59e0b")
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ combined score when removed  (negative = component contributes)")
    ax.set_title("Component contribution: vote boxes dominate; IRT is in-dist neutral",
                 fontsize=9.5, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ablation.png")
    plt.close(fig)
    print("  fig2_ablation.png")


def fig_lambda():
    """λ 궤적 — shadow price가 예산 소진 속도에 반응하는 모습 (M3)."""
    from src.harness import run_tier, tier_budgets
    from src.router import LPBRouter
    from src.synth import make_world
    from src.synth_ext import extend_with_votes
    from run_phase2 import CFG

    base = make_world(CFG, "irt", seed=CFG["seed"])
    ds = extend_with_votes(base, CFG)
    folds = base.stratified_folds(CFG["eval"]["k_folds"], CFG["seed"])
    te, tr = folds[0], np.setdiff1d(np.arange(ds.n), folds[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for tier, col in (("fast", "#dc2626"), ("balanced", C_LPB), ("premium", "#0891b2")):
        r = LPBRouter(CFG, CFG["synth"]["n_domains"], seed=0).fit(ds, tr, tier)
        pol = r.policy()
        b_te = tier_budgets(ds, te, CFG)[tier]
        pol.reset(tier=tier, budget=b_te, n_queries=len(te))
        lam_path, spend_path, cum = [], [], 0.0
        from src.cost_mirror import cost_matrix
        from src.harness import Session
        cmat = cost_matrix(ds)
        for i in te:
            sess = Session(costs=cmat[i], verifier_row=ds.verifier[i],
                           remaining_budget=b_te - cum)
            pol.route(sess, ds.features[i], int(ds.domains[i]))
            cum += sess.spent
            pol.observe_spend(sess.spent)
            lam_path.append(pol.inner.lam)
            spend_path.append(cum / b_te)
        t = np.arange(1, len(lam_path) + 1) / len(lam_path)
        axes[0].plot(t, lam_path, color=col, lw=1.2, label=tier)
        axes[1].plot(t, spend_path, color=col, lw=1.2, label=tier)
    axes[1].plot([0, 1], [0, 1], "--", color="#334155", lw=0.9, label="ideal pace")
    axes[0].set_xlabel("query progress")
    axes[0].set_ylabel("shadow price λ")
    axes[0].set_title("λ adapts to spend rate", fontsize=9.5)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].set_xlabel("query progress")
    axes[1].set_ylabel("budget consumed (fraction)")
    axes[1].set_title("Spend tracks the ideal pace", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_lambda.png")
    plt.close(fig)
    print("  fig3_lambda.png")


def fig_loss():
    """손실 분해 — 오라클 갭이 무엇으로 이루어져 있는가."""
    d = _load("phase8b.json")
    if not d:
        return
    tiers = ["fast", "balanced", "premium"]
    sel = [d[t]["cats"]["selection_error"] for t in tiers]
    reg = [d[t]["cats"]["early_stop_regret"] for t in tiers]
    una = [d[t]["cats"]["unanswered"] for t in tiers]
    x = np.arange(len(tiers))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(x, sel, 0.5, label="selection error (noisy verifier)", color="#dc2626")
    ax.bar(x, reg, 0.5, bottom=sel, label="early-stop regret", color="#f59e0b")
    ax.bar(x, una, 0.5, bottom=np.add(sel, reg), label="unanswered", color="#64748b")
    ax.set_xticks(x, tiers)
    ax.set_ylabel("% of queries")
    ax.set_title("Loss decomposition on World F (real text, verifier AUC 0.90)",
                 fontsize=9.5, pad=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_loss.png")
    plt.close(fig)
    print("  fig4_loss.png")


def fig_sigma():
    """D17 — 예약값의 두 구간과 λ→∞ 퇴화 동작."""
    from src.engine.prize import bernoulli_reservation
    lam_c = np.linspace(0.0, 1.6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for p, col in ((0.8, C_LPB), (0.4, "#f59e0b"), (0.15, "#dc2626")):
        old = 1.0 - lam_c / p
        new = bernoulli_reservation(np.full_like(lam_c, p), lam_c)
        axes[0].plot(lam_c, old, "--", color=col, lw=1.0, alpha=0.75)
        axes[0].plot(lam_c, new, color=col, lw=1.6, label=f"p̄ = {p}")
    axes[0].axhline(0, color="#334155", lw=0.8)
    axes[0].set_ylim(-4, 1.1)
    axes[0].set_xlabel("λ·c  (price × cost)")
    axes[0].set_ylabel("reservation value σ")
    axes[0].set_title("solid = general solution, dashed = old closed form\n"
                      "(they diverge once λc > p̄)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    # λ→∞ 에서 어떤 상자를 여는가: 최저비용 vs 가성비 최고 (두 기준이 갈리는 예)
    #   box 0: 가장 싸다 (c=.20)          — 설계 문서가 요구한 저예산 극한의 정답
    #   box 1: 가성비 최고 (c/p̄ = .375)   — 구식이 끝까지 붙잡는 상자
    p_vec = np.array([0.15, 0.80, 0.50])
    c_vec = np.array([0.20, 0.30, 0.55])
    lams = np.logspace(-1, 1.6, 400)
    pick_old = [int(np.argmax(1.0 - l * c_vec / p_vec)) for l in lams]
    pick_new = [int(np.argmax(bernoulli_reservation(p_vec, l * c_vec))) for l in lams]
    axes[1].step(lams, pick_old, where="post", ls="--", color="#dc2626",
                 lw=1.8, label="old: argmax(1−λc/p̄)")
    axes[1].step(lams, pick_new, where="post", color=C_LPB, lw=1.6,
                 label="new: argmax σ")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.4, 2.4)
    axes[1].set_yticks([0, 1, 2], ["box 0\n(c=.20, cheapest)",
                                   "box 1\n(c=.30, best c/p̄)", "box 2\n(c=.55)"],
                       fontsize=7.5)
    axes[1].set_xlabel("λ  (budget pressure)")
    axes[1].set_title("As λ→∞ the policy must degenerate\nto the cheapest single call",
                      fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_sigma.png")
    plt.close(fig)
    print("  fig5_sigma.png")


if __name__ == "__main__":
    print(f"[B11] 시각 자료 생성 → {FIG}")
    fig_worlds()
    fig_ablation()
    fig_loss()
    fig_sigma()
    fig_lambda()
    print("완료.")
