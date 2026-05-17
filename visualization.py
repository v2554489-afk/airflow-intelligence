import matplotlib.pyplot as plt

def plot_pollutants(df):

    # ✅ Safety check
    if df is None or df.empty:
        return None

    # ✅ Drop AQI safely
    pollutants = df.iloc[0].drop(labels=["AQI"], errors="ignore")

    # ✅ Ensure numeric values only
    pollutants = pollutants.apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0)

    fig, ax = plt.subplots(figsize=(12, 3.2))

    # modern neon palette
    colors = [
        "#00e5ff",  # cyan
        "#7c4dff",  # purple
        "#ff2e88",  # pink
        "#00ff9d",  # green
        "#ffd166",  # yellow
        "#ff5c5c"   # red
    ]

    bars = ax.bar(
        pollutants.index,
        pollutants.values,
        color=colors[:len(pollutants)],
        edgecolor="none"
    )

    # background
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")

    # title
    ax.set_title(
        "Pollutants Overview",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax.set_ylabel("Level", color="#94a3b8")

    # ticks
    ax.tick_params(axis='x', colors="#cbd5e1", rotation=0)
    ax.tick_params(axis='y', colors="#cbd5e1")

    # grid
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="white")

    # remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # value labels
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha='center',
            va='bottom',
            color="white",
            fontsize=9
        )

    plt.tight_layout()

    return fig
