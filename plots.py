# -*- coding: utf-8 -*-
"""
plots.py — LEC Tool
====================
All figures of the pipeline. Every ``plot_*`` function builds and returns a
matplotlib Figure without showing or saving it; ``save_figure`` writes it to
the run's output folder as ``<run_id>_<NN>_<slug>.png``.

Figure numbering is fixed so file names are stable whatever stages run:
  01 lec_curve                   05 drr_investment
  02 lec_analytical_vs_simulated 06 reduced_lec_curves
  03 catalogue_statistics        07 strategy_payouts_reduced
  04 strategy_payouts_base       08 horizon_loss_distribution
                                 09 cba_results (produced by cba.reports)

The matplotlib backend must be selected by the caller before this module is
imported (main.py switches to 'Agg' when figures are not shown).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def save_figure(fig, output_dir: Path, run_id: str, index: int, slug: str,
                dpi: int = 130, keep_open: bool = False) -> Path:
    """Save *fig* as PNG in *output_dir* and close it unless *keep_open*."""
    path = Path(output_dir) / f'{run_id}_{index:02d}_{slug}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if not keep_open:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_lec_curve(lec_result, lec_curve, aal, hybrid, event_loss_df):
    """Figure 01: empirical LEC with bootstrap band (+ hybrid) and annual losses."""
    if lec_result is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(lec_curve[:, 0], lec_curve[:, 1], label='Imported complete LEC')
        ax.set(xscale='log', yscale='log', xlabel='Economic loss [$MM]',
               ylabel='Annual frequency of exceedance', title='Loss Exceedance Curve (LEC)')
        ax.grid(True, which='both')
        ax.text(0.05, 0.8, f'AAL = ${aal:,.2f} MM', transform=ax.transAxes)
        ax.legend()
        fig.tight_layout()
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(lec_result['empirical'], lec_result['lambda_empirical'], 'k-',
            linewidth=1.5, label='Empirical LEC')
    ax.plot(lec_result['lec_mean'], lec_result['lambda_empirical'], 'b-',
            linewidth=2, label='Mean bootstrapped LEC')
    ax.fill_betweenx(lec_result['lambda_empirical'], lec_result['lec_p05'], lec_result['lec_p95'],
                     color=[0.8, 0.8, 1], alpha=0.5, label='90% CI')
    if hybrid:
        ax.plot(lec_curve[:, 0], lec_curve[:, 1], 'r--', linewidth=1.5, label='Hybrid LEC')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set_xlabel('Economic loss [$MM]')
    ax.set_ylabel('Annual frequency of exceedance')
    ax.set_title('Loss Exceedance Curve (LEC)')
    ax.text(0.05, 0.8, f'AAL = ${np.ceil(aal):,.0f} MM', transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.legend()

    yearly_loss = event_loss_df.groupby('year')['econ_loss'].sum().reset_index()
    ax = axes[1]
    ax.bar(yearly_loss['year'], yearly_loss['econ_loss'])
    ax.set_title('Historical economic loss per year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Economic loss [$MM]')
    ax.grid(True)

    fig.tight_layout()
    return fig


def plot_lec_vs_simulated(lec_curve, event_catalogue, catalogue_length, simulation_number):
    """Figure 02: analytical LEC vs the empirical LEC of the simulated events."""
    LS = lec_curve[:, 0]
    lambda_loss = lec_curve[:, 1]
    all_losses = [l for cat in event_catalogue for l in cat['losses']]

    hist, _ = np.histogram(all_losses, bins=np.append(LS, np.inf))
    lambda_simulated = np.flip(np.cumsum(np.flip(hist))) / (catalogue_length * simulation_number)

    fig, ax = plt.subplots()
    ax.plot(LS, lambda_loss, 'b-', linewidth=2, label='Original (analytical)')
    ax.plot(LS, lambda_simulated, 'r--', linewidth=1.5, label='Empirical (simulated)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Loss [$MM]')
    ax.set_ylabel('Annual frequency of exceedance')
    ax.set_title('LEC: analytical vs simulated')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', ls='--')
    ax.set_ylim(bottom=lambda_loss.min())
    fig.tight_layout()
    return fig


def plot_catalogue_statistics(synthetic_annual_df, catalogue_length, displayed_catalogue):
    """Figure 03: 2x2 grid of annual-loss statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    stat_labels = ['Std. deviation', 'Median', 'Maximum', 'Mean']

    data_cat = synthetic_annual_df.loc[displayed_catalogue].values
    data_all = synthetic_annual_df.values.flatten()

    axes[0, 0].bar(np.arange(1, catalogue_length + 1), data_cat)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Simulated annual economic loss ($MM)')
    axes[0, 0].set_title(f'Annual losses - catalogue {displayed_catalogue}')

    axes[1, 0].hist(data_all, bins=50)
    axes[1, 0].set_xlabel('Simulated annual economic loss ($MM)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of annual losses (all catalogues)')

    axes[0, 1].barh(stat_labels, [np.std(data_cat), np.median(data_cat), np.max(data_cat), np.mean(data_cat)])
    axes[0, 1].set_xlabel('Annual economic loss ($MM)')
    axes[0, 1].set_title(f'Statistics - catalogue {displayed_catalogue}')

    axes[1, 1].barh(stat_labels, [np.std(data_all), np.median(data_all), np.max(data_all), np.mean(data_all)])
    axes[1, 1].set_xlabel('Annual economic loss ($MM)')
    axes[1, 1].set_title('Statistics - all catalogues')

    fig.tight_layout()
    return fig


def plot_strategy_payouts(annual_df, payout_dfs, drm_configs, year_labels, displayed_catalogue,
                          title, ylim=None):
    """
    Figures 04/07: annual losses of one catalogue with stacked instrument payouts.

    Returns (fig, ylim) so the reduced-catalogue figure can reuse the base axis limit.
    """
    loss_vals = annual_df.loc[displayed_catalogue].values
    payouts = [df.loc[displayed_catalogue].values for df in payout_dfs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(year_labels, loss_vals, label='Fiscal losses')
    bottom = np.zeros(len(year_labels))
    for cfg, payout in zip(drm_configs, payouts):
        ax.bar(year_labels, payout, bottom=bottom, label=cfg['name'], hatch='//')
        bottom += payout
    for x, v in zip(year_labels, loss_vals):
        if v > 0:
            ax.text(x, v, f'${v:.1f}MM', ha='center', va='bottom', fontsize=9)
    if ylim is None:
        ylim = max(loss_vals.max(), bottom.max()) * 1.15 or 1.0
    ax.set_ylim(0, ylim)
    ax.set_xticks(year_labels)
    ax.set_xticklabels(year_labels)
    ax.set_xlabel('Year')
    ax.set_ylabel('Simulated annual fiscal loss ($MM)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ylim


def plot_drr_investment(year_labels, investment):
    """Figure 05: DRR investment per year."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(year_labels, investment, color='#1B5E79', label='DRR investment')
    for x, v in zip(year_labels, investment):
        ax.text(x, v, f'${v:.1f}MM' if v > 0 else '$-', ha='center', va='bottom', fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'${val:.1f}MM' if val > 0 else '$-'))
    ax.set_xticks(year_labels)
    ax.set_xticklabels(year_labels)
    ax.set_xlabel('Year')
    ax.set_title('DRR investment')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reduced_lec_curves(lec_curves_by_target):
    """Figure 06: family of ex-ante reduced LEC curves."""
    fig, ax = plt.subplots()
    ypos = 0.85
    for C_target in sorted(lec_curves_by_target.keys()):
        data = lec_curves_by_target[C_target]
        ax.semilogx(data['Loss'], data['Lambda'], label=f'C_target = {C_target:.1f}')
        ax.text(0.05, ypos, f'C = {C_target:.1f}: AAL = ${data["aal"]:,.2f} MM',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        ypos -= 0.08
    ax.set_xlabel('Loss [$MM]')
    ax.set_ylabel('Annual frequency of exceedance')
    ax.set_title('Reduced LEC curves (ex-ante risk reduction)')
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_horizon_loss_distribution(synthetic_annual_df, catalogue_length):
    """Figure 08: distribution of the cumulative loss over the simulation horizon."""
    totals = synthetic_annual_df.sum(axis=1)
    positive = totals[totals > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(positive) > 0 and positive.min() < positive.max():
        bins = np.geomspace(positive.min(), positive.max(), 60)
        ax.hist(positive, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_xscale('log')
    elif len(positive) > 0:
        ax.hist(positive, bins=10, color='steelblue', edgecolor='white', alpha=0.85)

    ax.axvline(totals.mean(), color='crimson', linewidth=2, label=f'Mean: ${totals.mean():,.1f} MM')
    ax.axvline(totals.median(), color='darkorange', linewidth=2, linestyle='--',
               label=f'Median: ${totals.median():,.1f} MM')
    ax.set_title(f'Distribution of cumulative losses over {catalogue_length} years')
    ax.set_xlabel('Total cumulative loss ($MM, log scale)')
    ax.set_ylabel('Number of simulations')
    ax.grid(True, which='both', axis='x', alpha=0.25)
    ax.legend()

    zero_count = int((totals == 0).sum())
    if zero_count:
        ax.text(0.02, 0.95, f'Simulations with zero loss: {zero_count:,}',
                transform=ax.transAxes, va='top')
    fig.tight_layout()
    return fig
