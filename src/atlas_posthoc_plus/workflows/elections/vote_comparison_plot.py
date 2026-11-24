"""Common plot styling for the adjusted ensemble and public vote comparisons."""

from atlas_posthoc_plus.paper import figure_legend, official_plan_style

POOLED_BOX_WIDTH = 0.30
_REFERENCE_MARKERS = ("o", "s", "^", "D", "P", "X")
_REFERENCE_COLORS = (
    "tab:purple", "tab:green", "tab:red", "tab:orange", "tab:blue", "tab:brown",
)


def ordered_rank_labels(n_districts: int) -> list[str]:
    labels: list[str] = []
    mid = n_districts // 2 + 1

    for rank in range(1, n_districts + 1):
        if rank == 1:
            labels.append("Most Rep")
        elif rank < mid:
            labels.append(f"{rank}")
        elif rank == mid:
            labels.append("Median")
        elif rank == n_districts:
            labels.append("Most Dem")
        else:
            labels.append(f"{n_districts - rank + 1}")

    return labels


def plot_reference_plan(ax, ranks, shares, label: str, index: int):
    line, = ax.plot(
        ranks,
        shares,
        **official_plan_style(
            label,
            marker=_REFERENCE_MARKERS[index % len(_REFERENCE_MARKERS)],
            color=_REFERENCE_COLORS[index % len(_REFERENCE_COLORS)],
        ),
        linewidth=1.6,
        markersize=4.5,
        label=label,
        zorder=3,
    )
    return line


def format_vote_comparison(ax, ranks, *, handles=None) -> None:
    n_districts = len(ranks)
    ax.set_xlabel("Districts ordered from most Republican to most Democratic")
    ax.set_ylabel("Democratic two-party vote share")
    ax.set_xlim(0.4, n_districts + 0.6)
    ax.set_xticks(ranks)
    ax.set_xticklabels(ordered_rank_labels(n_districts))
    figure_legend(ax, handles=handles, loc="lower right", frameon=True, fontsize=9)
